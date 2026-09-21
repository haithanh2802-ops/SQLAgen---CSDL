from decimal import Decimal

import pandas as pd

from text2sql_agent.ui_charts import ChartSpec, infer_chart_spec


def test_infers_review_score_chart_for_delivery_buckets():
    frame = pd.DataFrame(
        {
            "delay_bucket": ["On time", "1-3 days late", "4-7 days late"],
            "order_count": [1000, 200, 80],
            "avg_review_score": [Decimal("4.3"), Decimal("3.8"), Decimal("2.9")],
            "pct_one_star": [Decimal("0.08"), Decimal("0.15"), Decimal("0.32")],
        }
    )

    assert infer_chart_spec(frame) == ChartSpec(
        "bar", "delay_bucket", ("avg_review_score",)
    )


def test_valid_model_chart_metadata_takes_precedence():
    frame = pd.DataFrame({"month": ["Jan", "Feb"], "revenue": [10.0, 20.0]})

    assert infer_chart_spec(
        frame,
        requested_kind="line",
        requested_x="month",
        requested_y=["revenue"],
    ) == ChartSpec("line", "month", ("revenue",))


def test_single_row_result_does_not_create_a_chart():
    frame = pd.DataFrame({"total_orders": [100]})

    assert infer_chart_spec(frame) is None


def test_two_numeric_columns_create_a_scatter_chart():
    frame = pd.DataFrame(
        {
            "delivery_days": [1, 2, 4, 7],
            "review_score": [5.0, 4.5, 3.8, 2.9],
        }
    )

    assert infer_chart_spec(frame) == ChartSpec(
        "scatter", "delivery_days", ("review_score",)
    )
