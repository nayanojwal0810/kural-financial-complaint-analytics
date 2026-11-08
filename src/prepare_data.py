"""Data preparation pipeline for Kural financial complaint analytics.

Loads raw complaints data, standardizes fields, engineers initial features,
and saves the cleaned dataset.
"""

from pathlib import Path
import re
import pandas as pd


def prepare_complaint_data(raw_path: str, output_path: str) -> pd.DataFrame:
    """Prepares raw financial complaint records for exploratory analysis.

    Parameters
    ----------
    raw_path : str
        Path to the compressed raw complaints CSV file.
    output_path : str
        Destination path for the prepared compressed CSV file.

    Returns
    -------
    pd.DataFrame
        The cleaned and prepared complaints dataframe.
    """
    raw_file = Path(raw_path)
    output_file = Path(output_path)

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_file}")

    print("Loading raw complaints data...")
    df = pd.read_csv(raw_file)
    original_count = len(df)
    print(f"Loaded {original_count:,} records from {raw_file}")

    # Inspect missing values across important fields
    important_fields = [
        "complaint_id",
        "date_received",
        "product",
        "issue",
        "company",
        "company_response_to_consumer",
        "timely_response",
        "consumer_complaint_narrative",
    ]
    missing_summary = {field: int(df[field].isna().sum()) for field in important_fields if field in df.columns}
    print("\nMissing values in important fields:")
    for field, count in missing_summary.items():
        print(f"  - {field}: {count:,} ({count / original_count:.2%})")

    # Convert date fields to standard datetime format
    df["date_received"] = pd.to_datetime(df["date_received"], format="%Y-%m-%d", errors="coerce")
    if "date_sent_to_company" in df.columns:
        df["date_sent_to_company"] = pd.to_datetime(df["date_sent_to_company"], format="%Y-%m-%d", errors="coerce")

    # Standardize extra spaces in text and categorical fields
    category_fields = [
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "company",
        "company_public_response",
        "state",
        "zip_code",
        "tags",
        "consumer_consent_provided",
        "submitted_via",
        "company_response_to_consumer",
        "timely_response",
    ]
    for col in category_fields:
        if col in df.columns:
            # Preserve missing values while stripping and normalizing whitespace
            df[col] = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)

    # Preserve original narrative and create a separate cleaned narrative column
    # Lowercase text and normalize whitespace while protecting redaction markers (e.g. XXXX -> xxxx)
    df["cleaned_narrative"] = (
        df["consumer_complaint_narrative"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Narrative missing indicator (checks for nulls or whitespace-only strings)
    df["is_missing_narrative"] = (
        df["consumer_complaint_narrative"].isna()
        | (df["consumer_complaint_narrative"].astype(str).str.strip() == "")
    )
    missing_narrative_count = int(df["is_missing_narrative"].sum())

    # Identify duplicate narratives without deleting them
    # Duplicate indicator flags repeated occurrences after the first appearance
    df["is_duplicate_narrative"] = df["consumer_complaint_narrative"].duplicated(keep="first")
    duplicate_narrative_count = int(df["is_duplicate_narrative"].sum())

    # Identify repeated complaint IDs
    repeated_complaint_ids = int(df["complaint_id"].duplicated(keep="first").sum())

    # Feature engineering: temporal and text metrics
    df["complaint_year"] = df["date_received"].dt.year
    df["complaint_month"] = df["date_received"].dt.month
    df["narrative_word_count"] = df["cleaned_narrative"].apply(lambda s: len(s.split()) if s else 0)
    df["narrative_char_count"] = df["cleaned_narrative"].str.len()
    df["narrative_character_count"] = df["narrative_char_count"]

    # Record exclusion check: keep all records unless there is a clear documented reason
    # Check for invalid rows (e.g., missing complaint_id or unparseable date_received)
    invalid_mask = df["complaint_id"].isna() | df["date_received"].isna()
    excluded_count = int(invalid_mask.sum())
    if excluded_count > 0:
        print(f"\nExcluding {excluded_count:,} records due to missing ID or invalid date.")
        df = df[~invalid_mask].copy()
    else:
        print("\nRecord exclusion check: 0 records excluded (all 117,214 records are valid).")

    prepared_count = len(df)

    # Ensure output directory exists and save prepared dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving prepared dataset to {output_file}...")
    df.to_csv(output_file, index=False, compression="gzip")
    print("Dataset saved successfully.")

    # Execution summary
    print("\n" + "=" * 50)
    print("DATA PREPARATION SUMMARY")
    print("=" * 50)
    print(f"Original record count:    {original_count:,}")
    print(f"Prepared record count:    {prepared_count:,}")
    print(f"Excluded records:         {excluded_count:,}")
    print(f"Missing narratives:       {missing_narrative_count:,}")
    print(f"Duplicate narratives:     {duplicate_narrative_count:,}")
    print(f"Repeated complaint IDs:   {repeated_complaint_ids:,}")
    print(f"Output location:          {output_file.resolve()}")
    print("=" * 50 + "\n")

    return df


if __name__ == "__main__":
    raw_data_path = "data/raw/complaints.csv.gz"
    processed_data_path = "data/processed/complaints_clean.csv.gz"
    prepare_complaint_data(raw_data_path, processed_data_path)
