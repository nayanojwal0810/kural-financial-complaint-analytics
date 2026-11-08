"""Tests for statistical analysis tables in outputs/tables/."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tables_dir() -> Path:
    return Path("outputs/tables")


def test_product_timeliness_reconciliation(tables_dir: Path):
    df_timely = pd.read_csv(tables_dir / "product_timeliness_statistics.csv")

    # 1. Row totals: timely + untimely == total_complaints
    assert (df_timely["timely_count"] + df_timely["untimely_count"] == df_timely["total_complaints"]).all()

    # 2. Portfolio total equals 117,214
    assert df_timely["total_complaints"].sum() == 117214
    assert df_timely["timely_count"].sum() == 115136
    assert df_timely["untimely_count"].sum() == 2078

    # 3. Valid confidence intervals (lower <= percentage <= upper)
    assert (df_timely["ci_lower_pct"] <= df_timely["timely_percentage"]).all()
    assert (df_timely["timely_percentage"] <= df_timely["ci_upper_pct"]).all()
    assert (df_timely["ci_lower_pct"] < df_timely["ci_upper_pct"]).all()

    # 4. Percentage calculation matches counts within published rounding precision (0.01%)
    calculated_pct = (df_timely["timely_count"] / df_timely["total_complaints"] * 100).round(2)
    assert np.allclose(df_timely["timely_percentage"], calculated_pct, atol=0.01)


def test_product_recorded_relief_reconciliation(tables_dir: Path):
    df_relief = pd.read_csv(tables_dir / "product_recorded_relief.csv")

    # 1. Row totals: relief + no_relief == total_complaints
    assert (df_relief["relief_count"] + df_relief["no_relief_count"] == df_relief["total_complaints"]).all()

    # 2. Monetary + non-monetary == total relief
    assert (df_relief["monetary_count"] + df_relief["non_monetary_count"] == df_relief["relief_count"]).all()

    # 3. Portfolio totals
    assert df_relief["total_complaints"].sum() == 117214
    assert df_relief["relief_count"].sum() == 21555
    assert df_relief["monetary_count"].sum() == 5118
    assert df_relief["non_monetary_count"].sum() == 16437

    # 4. Valid confidence intervals
    assert (df_relief["ci_lower_pct"] <= df_relief["relief_percentage"]).all()
    assert (df_relief["relief_percentage"] <= df_relief["ci_upper_pct"]).all()
    assert (df_relief["ci_lower_pct"] < df_relief["ci_upper_pct"]).all()

    # 5. Percentages agree within published rounding precision (0.01%)
    calc_relief_pct = (df_relief["relief_count"] / df_relief["total_complaints"] * 100).round(2)
    assert np.allclose(df_relief["relief_percentage"], calc_relief_pct, atol=0.01)


def test_product_counts_reconcile_across_tables(tables_dir: Path):
    df_prod = pd.read_csv(tables_dir / "complaints_by_product.csv")
    df_timely = pd.read_csv(tables_dir / "product_timeliness_statistics.csv")
    df_relief = pd.read_csv(tables_dir / "product_recorded_relief.csv")
    df_narr = pd.read_csv(tables_dir / "narrative_length_by_product.csv")

    # Reconcile product totals across all 4 product tables
    counts_prod = df_prod.set_index("product")["complaint_count"].sort_index()
    counts_timely = df_timely.set_index("product")["total_complaints"].sort_index()
    counts_relief = df_relief.set_index("product")["total_complaints"].sort_index()
    counts_narr = df_narr.set_index("product")["total_complaints"].sort_index()

    pd.testing.assert_series_equal(counts_prod, counts_timely, check_names=False)
    pd.testing.assert_series_equal(counts_prod, counts_relief, check_names=False)
    pd.testing.assert_series_equal(counts_prod, counts_narr, check_names=False)
