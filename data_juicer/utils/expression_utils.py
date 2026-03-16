from typing import Union
from ray.data.expressions import Expr, col


def build_range_expr(
    column: Union[str, Expr],
    min_val: float = None,
    max_val: float = None,
    min_closed_interval: bool = True,
    max_closed_interval: bool = True,
    reversed_range: bool = False,
):
    """
    根据给定的区间配置构建 Ray Data 的表达式 (Predicate)。

    :param column: 统计信息对应的列名（例如：`StatsKeys.avg_line_length`）或已经构建好的 Ray 表达式
    :param min_val: 区间下界
    :param max_val: 区间上界
    :param min_closed_interval: 是否为左闭区间 (>=)
    :param max_closed_interval: 是否为右闭区间 (<=)
    :param reversed_range: 是否取反 (将保留区间外的值)
    :return: ray.data.expressions.Expression
    """
    expr = None
    c = col(column) if isinstance(column, str) else column

    # 1. 处理 min_val
    if min_val is not None:
        if min_closed_interval:
            expr = c >= min_val
        else:
            expr = c > min_val

    # 2. 处理 max_val
    if max_val is not None:
        if max_closed_interval:
            max_expr = c <= max_val
        else:
            max_expr = c < max_val

        if expr is not None:
            expr = expr & max_expr
        else:
            expr = max_expr

    # 3. 处理无效输入
    if expr is None:
        raise ValueError("Both min_val and max_val cannot be None.")

    # 4. 根据 reversed_range 反转逻辑
    if reversed_range:
        expr = ~expr

    return expr


def build_in_list_expr(column: Union[str, Expr], target_list: list, keep_in_list: bool = True):
    """
    支持在列表中 (isin) 的判定
    """
    c = col(column) if isinstance(column, str) else column
    expr = c.isin(target_list)
    if not keep_in_list:
        expr = ~expr
    return expr
