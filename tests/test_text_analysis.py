"""Tests for NLP text preparation and topic modeling in src/text_analysis.py."""

import pandas as pd
import pytest

from src.text_analysis import (
    assign_topics,
    extract_topic_keywords,
    fit_topic_model,
    prepare_nlp_text,
)


def test_prepare_nlp_text_removes_redactions_and_preserves_words():
    # 1. Redaction placeholders removed
    raw_sample = "Reported to XXXX on XX/XX/XXXX and XXXX-XXXX regarding balance."
    cleaned = prepare_nlp_text(raw_sample)

    assert "xxxx" not in cleaned
    assert "xx/xx/xxxx" not in cleaned
    assert "balance" in cleaned
    assert "reported" in cleaned

    # 2. Meaningful words and numbers preserved
    financial_sample = "Credit card interest fee was charged on 12/05/2019 without disclosure."
    cleaned_fin = prepare_nlp_text(financial_sample)
    for term in ["credit", "card", "interest", "fee", "charged", "disclosure"]:
        assert term in cleaned_fin

    # 3. Words containing letter 'x' preserved (not stripped)
    extra_sample = "I asked for an extra tax explanation."
    cleaned_extra = prepare_nlp_text(extra_sample)
    assert "extra" in cleaned_extra
    assert "tax" in cleaned_extra
    assert "explanation" in cleaned_extra

    # 4. Empty and missing text handled safely
    assert prepare_nlp_text("") == ""
    assert prepare_nlp_text("   ") == ""
    assert prepare_nlp_text(None) == ""


def test_topic_modeling_pipeline_on_sample():
    # Small test dataset with distinct financial themes
    sample_texts = [
        "Debt collection agency calling repeatedly about medical debt not owed.",
        "Collection agency calls multiple times a day demanding payment for invalid debt.",
        "Unauthorized inquiry on credit report from auto lender without permission.",
        "Credit reporting bureau has fraudulent inquiry that I did not authorize.",
        "Checking account overdraft fees charged improperly by bank teller.",
        "Bank checking account closed with funds frozen and check held.",
        "Mortgage servicer failed to apply monthly escrow payment on home loan.",
        "Home mortgage payment late fee added despite paying on time to servicer.",
    ]

    df_sample = pd.DataFrame({
        "complaint_id": range(1, len(sample_texts) + 1),
        "cleaned_narrative": sample_texts,
    })

    # Prepare distinct text series
    distinct_series = pd.Series(df_sample["cleaned_narrative"].apply(prepare_nlp_text).unique())

    # Fit a 3-topic NMF model with lower min_df for testing
    tfidf, nmf = fit_topic_model(distinct_series, n_components=3, random_state=42, min_df=1)

    # Verify topic keywords extracted
    keywords = extract_topic_keywords(tfidf, nmf, n_words=5)
    assert len(keywords) == 3
    assert all(len(words) == 5 for words in keywords.values())

    # Assign topics to sample
    df_assigned = assign_topics(df_sample, tfidf, nmf, text_col="cleaned_narrative")

    # 1. Topic IDs in expected format (topic_01, topic_02, topic_03)
    valid_topic_ids = {"topic_01", "topic_02", "topic_03"}
    assert df_assigned["topic_id"].isin(valid_topic_ids).all()

    # 2. Weight in valid non-negative range
    assert (df_assigned["topic_weight"] >= 0.0).all()


def test_repeated_identical_text_consistency():
    # Identical texts must receive identical topic assignments
    training_texts = pd.Series([
        "Late fee charged improperly on credit card account balance.",
        "Debt collector harassed me by telephone at work.",
        "Mortgage escrow shortage calculation error on closing.",
    ])
    tfidf, nmf = fit_topic_model(training_texts, n_components=2, random_state=42, min_df=1)

    df_eval = pd.DataFrame({
        "complaint_id": [1, 2, 3],
        "cleaned_narrative": [
            "Debt collector harassed me by telephone at work.",
            "Late fee charged improperly on credit card account balance.",
            "Debt collector harassed me by telephone at work.",  # identical to row 1
        ]
    })
    assigned = assign_topics(df_eval, tfidf, nmf, text_col="cleaned_narrative")
    assert assigned.loc[0, "topic_id"] == assigned.loc[2, "topic_id"]
    assert assigned.loc[0, "topic_weight"] == assigned.loc[2, "topic_weight"]
