"""
Lineage Logging Mixin - Adds lineage tracking capabilities to executors.

Similar to EventLoggingMixin, this mixin is added to executor class hierarchies
to automatically capture pipeline and operator lineage events.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Set

from loguru import logger

from data_juicer.core.lineage.adapter import LineageAdapter
from data_juicer.core.lineage.config import LineageConfig


class LineageLoggingMixin:
    """Mixin to add lineage tracking capabilities to any executor.

    Usage in executor class::

        class DefaultExecutor(ExecutorBase, DAGExecutionMixin,
                              EventLoggingMixin, LineageLoggingMixin):
            def __init__(self, cfg):
                super().__init__(cfg)
                LineageLoggingMixin.__init__(self)

    The mixin provides:
    - Pipeline-level lineage: on_pipeline_start / complete / fail
    - Operator-level lineage: on_op_start / complete / fail
    - Automatic column diff tracking for Level 2 lineage
    """

    def __init__(self, *args, **kwargs):
        """Initialize the lineage logging mixin."""
        if not hasattr(self, '_lineage_adapter'):
            self._setup_lineage()

    def _setup_lineage(self) -> None:
        """Setup lineage tracking from executor configuration."""
        self._lineage_adapter: Optional[LineageAdapter] = None

        # Build LineageConfig from executor cfg
        cfg = getattr(self, 'cfg', None)
        if cfg is None:
            return

        lineage_cfg_data = getattr(cfg, 'lineage', None)
        if lineage_cfg_data is None:
            # Lineage not configured — disabled by default
            return

        # Convert Namespace to dict if needed
        if hasattr(lineage_cfg_data, '__dict__'):
            lineage_cfg_data = dict(lineage_cfg_data)

        try:
            lineage_config = LineageConfig.from_dict(lineage_cfg_data)
        except Exception as e:
            logger.warning(f'Failed to parse lineage config: {e}')
            return

        if not lineage_config.enabled:
            return

        # Set default file transport path if not provided
        if (lineage_config.transport.type == 'file'
                and lineage_config.transport.log_file_path is None):
            work_dir = getattr(self, 'work_dir', None) or '.'
            lineage_dir = os.path.join(work_dir, 'lineage')
            os.makedirs(lineage_dir, exist_ok=True)
            lineage_config.transport.log_file_path = os.path.join(
                lineage_dir, 'events.jsonl'
            )

        self._lineage_adapter = LineageAdapter(lineage_config)

    @property
    def lineage_enabled(self) -> bool:
        """Check if lineage tracking is enabled."""
        return (self._lineage_adapter is not None
                and self._lineage_adapter.enabled)

    def log_lineage_pipeline_start(
        self,
        input_dataset: Optional[str] = None,
        num_operators: Optional[int] = None,
    ) -> None:
        """Log pipeline START lineage event."""
        if not self.lineage_enabled:
            return

        job_name = self._get_lineage_job_name()
        self._lineage_adapter.on_pipeline_start(
            job_name=job_name,
            input_dataset=input_dataset,
            num_operators=num_operators,
        )

    def log_lineage_pipeline_complete(
        self,
        output_dataset: Optional[str] = None,
        row_count: Optional[int] = None,
    ) -> None:
        """Log pipeline COMPLETE lineage event."""
        if not self.lineage_enabled:
            return

        self._lineage_adapter.on_pipeline_complete(
            output_dataset=output_dataset,
            row_count=row_count,
        )

    def log_lineage_pipeline_fail(self, error: Exception) -> None:
        """Log pipeline FAIL lineage event."""
        if not self.lineage_enabled:
            return

        self._lineage_adapter.on_pipeline_fail(error)

    # ------------------------------------------------------------------ #
    #  Operator-Level Hooks
    # ------------------------------------------------------------------ #

    def log_lineage_op_start(
        self,
        op: Any,
        op_index: int,
        columns_before: Optional[Set[str]] = None,
        row_count_before: Optional[int] = None,
    ) -> None:
        """Log operator START lineage event."""
        if not self.lineage_enabled:
            return

        self._lineage_adapter.on_op_start(
            op=op,
            op_index=op_index,
            columns_before=columns_before,
            row_count_before=row_count_before,
        )

    def log_lineage_op_complete(
        self,
        op: Any,
        op_index: int,
        columns_before: Optional[Set[str]] = None,
        columns_after: Optional[Set[str]] = None,
        row_count_before: Optional[int] = None,
        row_count_after: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> None:
        """Log operator COMPLETE lineage event."""
        if not self.lineage_enabled:
            return

        self._lineage_adapter.on_op_complete(
            op=op,
            op_index=op_index,
            columns_before=columns_before,
            columns_after=columns_after,
            row_count_before=row_count_before,
            row_count_after=row_count_after,
            duration=duration,
        )

    def log_lineage_op_fail(
        self,
        op: Any,
        op_index: int,
        error: Exception,
    ) -> None:
        """Log operator FAIL lineage event."""
        if not self.lineage_enabled:
            return

        self._lineage_adapter.on_op_fail(
            op=op,
            op_index=op_index,
            error=error,
        )

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def _get_lineage_job_name(self) -> str:
        """Get a meaningful job name from config."""
        cfg = getattr(self, 'cfg', None)
        if cfg is None:
            return 'unknown'

        # Try config file name
        config_file = getattr(cfg, 'config', None)
        if config_file:
            return os.path.splitext(os.path.basename(config_file))[0]

        # Fall back to project name
        return getattr(cfg, 'project_name', 'data-juicer')

    def close_lineage(self) -> None:
        """Close lineage adapter and release resources."""
        if self._lineage_adapter is not None:
            self._lineage_adapter.close()
            self._lineage_adapter = None
