from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

ChartKind = Literal["bar", "line", "area", "scatter"]


@dataclass(frozen=True)
class ChartSpec:
    kind: ChartKind
    x: str
    y: tuple[str, ...]


def infer_chart_spec(
    frame: pd.DataFrame,
    *,
    requested_kind: str = "none",
    requested_x: str | None = None,
    requested_y: list[str] | None = None,
) -> ChartSpec | None:
    """Use valid model chart metadata, or infer one conservative chart."""
    if frame.empty or len(frame.index) < 2:
        return None

    numeric = [column for column in frame.columns if _is_numeric_series(frame[column])]
    requested_numeric = [column for column in (requested_y or []) if column in numeric]
    if (
        requested_kind in {"bar", "line", "area", "scatter"}
        and requested_x in frame.columns
        and requested_numeric
    ):
        return ChartSpec(requested_kind, requested_x, tuple(requested_numeric))

    dimensions = [
        column
        for column in frame.columns
        if column not in numeric and 1 < frame[column].nunique(dropna=True) <= 40
    ]
    if not dimensions:
        if len(numeric) < 2:
            return None
        x_column = numeric[0]
        y_column = min(numeric[1:], key=_metric_priority)
        return ChartSpec("scatter", x_column, (y_column,))

    if not numeric:
        return None

    x_column = dimensions[0]
    y_column = min(numeric, key=_metric_priority)
    x_name = x_column.casefold()
    is_time_axis = is_datetime64_any_dtype(frame[x_column]) or any(
        token in x_name for token in ("date", "month", "year", "week", "day")
    )
    return ChartSpec("line" if is_time_axis else "bar", x_column, (y_column,))


def _metric_priority(column: str) -> tuple[int, int]:
    name = column.casefold()
    priorities = (
        ("avg", "average", "score"),
        ("rate", "pct", "percent", "ratio", "share"),
        ("revenue", "value", "amount", "price", "cost"),
        ("count", "total", "qty", "quantity"),
    )
    for priority, tokens in enumerate(priorities):
        if any(token in name for token in tokens):
            return priority, len(name)
    return len(priorities), len(name)


def _is_numeric_series(series: pd.Series) -> bool:
    if is_numeric_dtype(series):
        return True
    populated = series.dropna()
    if populated.empty:
        return False
    return pd.to_numeric(populated, errors="coerce").notna().all()
