"""
Lineage Transports - Pluggable event emission backends.

Inspired by OpenLineage Transport architecture.
"""

from data_juicer.core.lineage.transports.base import BaseTransport
from data_juicer.core.lineage.transports.composite import CompositeTransport
from data_juicer.core.lineage.transports.console import ConsoleTransport
from data_juicer.core.lineage.transports.factory import create_transport
from data_juicer.core.lineage.transports.file import FileTransport
from data_juicer.core.lineage.transports.http import HttpTransport

__all__ = [
    "BaseTransport",
    "ConsoleTransport",
    "FileTransport",
    "HttpTransport",
    "CompositeTransport",
    "create_transport",
]
