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
    Construct a Ray Data expression (Predicate) based on the given interval configuration.

    :param column: The column name corresponding to the statistical information (for example: `StatsKeys.avg_line_length`) or a pre-built Ray expression
    :param min_val: The lower bound of the interval
    :param max_val: The upper bound of the interval
    :param min_closed_interval: Whether the interval is left-closed (>=)
    :param max_closed_interval: Whether the interval is right-closed (<=)
    :param reversed_range: Whether to reverse the range (keep values outside the interval)
    :return: ray.data.expressions.Expression
    """
    expr = None
    c = col(column) if isinstance(column, str) else column

    if min_val is not None:
        if min_closed_interval:
            expr = c >= min_val
        else:
            expr = c > min_val

    if max_val is not None:
        if max_closed_interval:
            max_expr = c <= max_val
        else:
            max_expr = c < max_val

        if expr is not None:
            expr = expr & max_expr
        else:
            expr = max_expr

    if expr is None:
        raise ValueError("Both min_val and max_val cannot be None.")

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
