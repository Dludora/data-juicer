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
        self.strict_sdk = False

        self._initialize()

    def _initialize(self) -> None:
        lineage_cfg = ConfigAccessor.get(self.cfg, "lineage", {})
        self.enabled = bool(ConfigAccessor.get(lineage_cfg, "enabled", False))
        self.fail_silently = bool(ConfigAccessor.get(lineage_cfg, "fail_silently", True))
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

        transport_cfg = ConfigAccessor.get(lineage_cfg, "transport", None)
        transport_type = ConfigAccessor.get(lineage_cfg, "transport_type", "http")
        endpoint = ConfigAccessor.get(lineage_cfg, "endpoint", None)
        timeout = float(ConfigAccessor.get(lineage_cfg, "timeout", 3.0))
        api_key = ConfigAccessor.get(lineage_cfg, "api_key", None)
        retry_count = int(ConfigAccessor.get(lineage_cfg, "retry_count", 2))
        retry_backoff_seconds = float(ConfigAccessor.get(lineage_cfg, "retry_backoff_seconds", 1.0))

        client_config = None
        if transport_cfg:
            # Preferred path: pass native OpenLineage transport config directly.
            client_config = {"transport": transport_cfg}
        elif endpoint:
            # Backward-compatible shortcut config.
            transport_config = {
                "type": transport_type,
                "url": endpoint,
                "timeout": timeout,
                "retry": {
                    "total": retry_count,
                    "backoff_factor": retry_backoff_seconds,
                },
            }
            if api_key:
                transport_config["auth"] = {"type": "api_key", "apiKey": api_key}
            client_config = {"transport": transport_config}

        try:
            self.client = OpenLineageClient(config=client_config)
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

        try:
            self.client.emit(event)
        except Exception as e:
            if self.fail_silently:
                logger.warning(f"OpenLineage emit failed and ignored: {e}")
                return
            raise RuntimeError("Failed to emit OpenLineage event") from e
