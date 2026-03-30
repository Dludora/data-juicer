import time
from typing import Any

from loguru import logger

from data_juicer.utils.config_utils import ConfigAccessor


class OpenLineageTransport:
    """Thin fault-tolerant transport wrapper around OpenLineageClient."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.client = None
        self.enabled = False
        self.fail_silently = True
        self.retry_count = 2
        self.retry_backoff_seconds = 1.0
        self.strict_sdk = False

        self._initialize()

    def _initialize(self) -> None:
        lineage_cfg = ConfigAccessor.get(self.cfg, "lineage", {})
        self.enabled = bool(ConfigAccessor.get(lineage_cfg, "enabled", False))
        self.fail_silently = bool(ConfigAccessor.get(lineage_cfg, "fail_silently", True))
        self.retry_count = int(ConfigAccessor.get(lineage_cfg, "retry_count", 2))
        self.retry_backoff_seconds = float(ConfigAccessor.get(lineage_cfg, "retry_backoff_seconds", 1.0))
        self.strict_sdk = bool(ConfigAccessor.get(lineage_cfg, "strict_sdk", False))

        if not self.enabled:
            return

        try:
            from openlineage.client import OpenLineageClient
        except Exception as e:
            if self.strict_sdk:
                raise RuntimeError("OpenLineage SDK is required but unavailable") from e
            logger.warning(f"OpenLineage SDK not available, lineage disabled: {e}")
            self.enabled = False
            return

        transport_type = ConfigAccessor.get(lineage_cfg, "transport_type", "http")
        endpoint = ConfigAccessor.get(lineage_cfg, "endpoint", None)
        timeout = float(ConfigAccessor.get(lineage_cfg, "timeout", 3.0))
        api_key = ConfigAccessor.get(lineage_cfg, "api_key", None)

        client_config = {}
        if endpoint:
            transport_config = {"type": transport_type, "url": endpoint, "timeout": timeout}
            if api_key:
                transport_config["auth"] = {"type": "api_key", "apiKey": api_key}
            client_config["transport"] = transport_config

        try:
            self.client = OpenLineageClient(config=client_config or None)
            self.enabled = True
        except Exception as e:
            if self.fail_silently:
                logger.warning(f"Failed to initialize OpenLineage client, lineage disabled: {e}")
                self.enabled = False
                return
            raise

    def emit(self, event: Any) -> None:
        if not self.enabled or self.client is None:
            return

        last_error = None
        for attempt in range(self.retry_count + 1):
            try:
                self.client.emit(event)
                return
            except Exception as e:
                last_error = e
                if attempt < self.retry_count:
                    sleep_seconds = self.retry_backoff_seconds * (2**attempt)
                    logger.warning(
                        f"OpenLineage emit failed (attempt {attempt + 1}/{self.retry_count + 1}): {e}. "
                        f"Retry in {sleep_seconds:.1f}s"
                    )
                    time.sleep(sleep_seconds)

        if self.fail_silently:
            logger.warning(f"OpenLineage emit dropped after retries: {last_error}")
            return

        raise RuntimeError("Failed to emit OpenLineage event") from last_error
