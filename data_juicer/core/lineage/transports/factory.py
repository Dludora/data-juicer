"""
Transport Factory - Creates transport instances from configuration.

Supports built-in transport types and custom transports via fully-qualified
class names (following OpenLineage convention).
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

from loguru import logger

from data_juicer.core.lineage.transports.base import BaseTransport

# Registry of built-in transport types
_TRANSPORT_REGISTRY: Dict[str, str] = {
    "console": "data_juicer.core.lineage.transports.console.ConsoleTransport",
    "file": "data_juicer.core.lineage.transports.file.FileTransport",
    "http": "data_juicer.core.lineage.transports.http.HttpTransport",
    "composite": "data_juicer.core.lineage.transports.composite.CompositeTransport",
}


def create_transport(config: Dict[str, Any]) -> BaseTransport:
    """Create a transport instance from a configuration dictionary.

    The 'type' field selects the transport class:
    - Built-in: 'console', 'file', 'http', 'composite'
    - Custom: fully-qualified class name (e.g., 'my_module.MyTransport')

    Args:
        config: Transport configuration dictionary with at least a 'type' key.

    Returns:
        Configured BaseTransport instance.

    Raises:
        ValueError: If type is missing or class cannot be found.
    """
    transport_type = config.get("type", "console")

    # Look up in built-in registry
    if transport_type in _TRANSPORT_REGISTRY:
        class_path = _TRANSPORT_REGISTRY[transport_type]
    else:
        # Assume it's a fully-qualified class name
        class_path = transport_type

    # Import the class
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        transport_cls = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ValueError(
            f"Cannot load transport class '{class_path}' for type '{transport_type}': {e}"
        ) from e

    if not issubclass(transport_cls, BaseTransport):
        raise ValueError(
            f"Transport class '{class_path}' must be a subclass of BaseTransport"
        )

    return transport_cls.from_config(config)


def register_transport(name: str, class_path: str) -> None:
    """Register a custom transport type.

    Args:
        name: Short name for the transport (e.g., 'kafka').
        class_path: Fully-qualified class path (e.g., 'my_module.KafkaTransport').
    """
    _TRANSPORT_REGISTRY[name] = class_path
    logger.debug(f"Registered lineage transport: {name} -> {class_path}")
