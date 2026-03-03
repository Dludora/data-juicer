"""
HTTP Transport - Emits lineage events via HTTP POST.

Sends events as JSON to an HTTP endpoint (e.g., Marquez, custom backend).
Supports authentication, compression, and retry.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger

from data_juicer.core.lineage.transports.base import BaseTransport, TransportError

if TYPE_CHECKING:
    from data_juicer.core.lineage.events import LineageEvent


class HttpTransport(BaseTransport):
    """Emits lineage events via synchronous HTTP POST.

    Config:
        - url (str): Base URL. Required.
        - endpoint (str): API path appended to url. Default: 'api/v1/lineage'.
        - timeout (float): Request timeout in seconds. Default: 5.0.
        - verify (bool): Verify TLS certs. Default: True.
        - auth (dict): Auth config, e.g. {'type': 'api_key', 'apiKey': '...'}.
        - compression (str): 'gzip' or None. Default: None.
        - retry (dict): Retry config. Default: {'total': 3, 'backoff_factor': 0.3}.
    """

    def __init__(
        self,
        url: str,
        endpoint: str = "api/v1/lineage",
        timeout: float = 5.0,
        verify: bool = True,
        auth: Optional[Dict[str, str]] = None,
        compression: Optional[str] = None,
        retry: Optional[Dict[str, Any]] = None,
    ):
        self._url = url.rstrip("/")
        self._endpoint = endpoint.lstrip("/")
        self._full_url = f"{self._url}/{self._endpoint}"
        self._timeout = timeout
        self._verify = verify
        self._auth = auth or {}
        self._compression = compression
        self._retry = retry or {"total": 3, "backoff_factor": 0.3, "status_forcelist": [500, 502, 503, 504]}

        self._session = None

    def _get_session(self):
        """Lazy-initialize HTTP session."""
        if self._session is not None:
            return self._session

        try:
            import httpx
        except ImportError:
            raise ImportError(
                "HttpTransport requires the 'httpx' package. "
                "Install it with: pip install httpx"
            )

        # Build headers
        headers = {"Content-Type": "application/json"}
        auth_type = self._auth.get("type")
        if auth_type == "api_key":
            api_key = self._auth.get("apiKey", "")
            headers["Authorization"] = f"Bearer {api_key}"

        self._session = httpx.Client(
            timeout=self._timeout,
            verify=self._verify,
            headers=headers,
        )
        return self._session

    def emit(self, event: "LineageEvent") -> None:
        """Send event as HTTP POST."""
        event_dict = event.to_dict()
        event_json = json.dumps(event_dict, ensure_ascii=False, default=str)

        # Prepare payload
        data = event_json.encode("utf-8")
        headers = {}
        if self._compression == "gzip":
            import gzip

            data = gzip.compress(data)
            headers["Content-Encoding"] = "gzip"

        # Retry logic
        max_retries = self._retry.get("total", 3)
        backoff_factor = self._retry.get("backoff_factor", 0.3)
        status_forcelist = self._retry.get("status_forcelist", [500, 502, 503, 504])

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                session = self._get_session()
                response = session.post(
                    self._full_url,
                    content=data,
                    headers=headers,
                )

                if response.status_code in status_forcelist:
                    raise TransportError(
                        f"Server returned {response.status_code}",
                        transport_type="http",
                    )

                if response.status_code >= 400:
                    raise TransportError(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        transport_type="http",
                    )

                return  # Success

            except TransportError:
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    sleep_time = backoff_factor * (2**attempt)
                    logger.debug(f"HTTP transport retry {attempt + 1}/{max_retries}, sleeping {sleep_time:.1f}s: {e}")
                    time.sleep(sleep_time)

        raise TransportError(
            f"Failed after {max_retries + 1} attempts: {last_error}",
            transport_type="http",
            cause=last_error,
        )

    def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HttpTransport":
        """Create HttpTransport from config dict."""
        url = config.get("url")
        if not url:
            raise ValueError("HttpTransport requires 'url' in config")

        return cls(
            url=url,
            endpoint=config.get("endpoint", "api/v1/lineage"),
            timeout=config.get("timeout", 5.0),
            verify=config.get("verify", True),
            auth=config.get("auth"),
            compression=config.get("compression"),
            retry=config.get("retry"),
        )
