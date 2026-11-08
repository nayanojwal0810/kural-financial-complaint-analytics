"""Tests for data preparation pipeline in src/prepare_data.py."""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.prepare_data import prepare_complaint_data


def test_prepare_data_with_sample(tmp_path: Path):
    # Create a small sample CSV mimicking raw complaints
    sample_data = pd.DataFrame({
        "complaint_id": [101, 102, 103, 104, 105],
        "date_received": ["2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05"],
        "product": [
            "Credit card",
            "Checking account ",
            "Credit card",
            "Mortgage",
            "Debt collection",
        ],
        "sub_product": ["General", "Personal", "General", "Conventional", "Medical"],
        "issue": ["Billing", "Fees", "Billing", "Closing", "Collection"],
        "sub_issue": ["Late fee", "Overdraft", "Late fee", "Delay", "Verification"],
        "consumer_complaint_narrative": [
            "Payment was not recorded on time by company.",
            np.nan,  # Missing narrative test
            "Payment was not recorded on time by company.",  # Duplicate narrative test
            "   Mortgage escrow   shortage error with extra   spaces.  ",
            "Debt collector called repeatedly at work.",
        ],
        "company_public_response": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "company": ["Bank A", "Bank B", "Bank A", "Bank C", "Agency D"],
        "state": ["CA", "NY", "CA", "TX", "FL"],
        "zip_code": ["90001", "10001", "90001", "75001", "33101"],
        "tags": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "consumer_consent_provided": ["Consent provided"] * 5,
        "submitted_via": ["Web"] * 5,
        "date_sent_to_company": ["2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06"],
        "company_response_to_consumer": [
            "Closed with explanation",
            "Closed with monetary relief",
            "Closed with non-monetary relief",
            "Untimely response",
            "Closed with explanation",
        ],
        "timely_response": ["Yes", "Yes", "Yes", "No", "Yes"],
        "consumer_disputed": [np.nan] * 5,
    })

    raw_path = tmp_path / "sample_raw.csv.gz"
    sample_data.to_csv(raw_path, index=False, compression="gzip")
    output_path = tmp_path / "sample_clean.csv.gz"

    df_clean = prepare_complaint_data(str(raw_path), str(output_path))

    # 1. Original records preserved
    assert len(df_clean) == 5
    assert set(df_clean["complaint_id"]) == {101, 102, 103, 104, 105}

    # 2. Missing narrative handled safely (never literal "nan")
    missing_row = df_clean.loc[df_clean["complaint_id"] == 102].iloc[0]
    assert missing_row["is_missing_narrative"] == True
    assert missing_row["cleaned_narrative"] == ""
    assert missing_row["narrative_word_count"] == 0
    assert missing_row["narrative_char_count"] == 0

    # 3. Text normalization: extra spaces collapsed and stripped
    space_row = df_clean.loc[df_clean["complaint_id"] == 104].iloc[0]
    assert space_row["cleaned_narrative"] == "mortgage escrow shortage error with extra spaces."
    assert space_row["narrative_word_count"] == 7

    # 4. Duplicate narrative indicator
    # Row 101 is first occurrence -> False; Row 103 is duplicate copy -> True
    dup_first = df_clean.loc[df_clean["complaint_id"] == 101, "is_duplicate_narrative"].values[0]
    dup_second = df_clean.loc[df_clean["complaint_id"] == 103, "is_duplicate_narrative"].values[0]
    assert dup_first == False
    assert dup_second == True

    # 5. Recorded-relief feature matching defined company response categories
    # True for "Closed with monetary relief" and "Closed with non-monetary relief"
    relief_series = df_clean.set_index("complaint_id")["has_recorded_relief"]
    assert relief_series[101] == False  # Closed with explanation
    assert relief_series[102] == True   # Closed with monetary relief
    assert relief_series[103] == True   # Closed with non-monetary relief
    assert relief_series[104] == False  # Untimely response
    assert relief_series[105] == False  # Closed with explanation
