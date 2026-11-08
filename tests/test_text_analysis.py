"""Tests for NLP text preparation and topic modeling in src/text_analysis.py."""

import pandas as pd
import pytest

from src.text_analysis import (
    assign_topics,
    extract_topic_keywords,
    fit_topic_model,
    prepare_nlp_text,
)


def test_prepare_nlp_text_removes_placeholders_and_preserves_financial_terms():
    # 1. Redaction placeholders and masking patterns removed
    raw_sample = "Reported to XXXX on XX/XX/XXXX and XXXX-XXXX regarding balance."
    cleaned = prepare_nlp_text(raw_sample)
    assert "xxxx" not in cleaned
    assert "xx/xx/xxxx" not in cleaned
    assert "balance" in cleaned
    assert "reported" in cleaned

    # 2. Zero-only placeholder tokens and CFPB amount redactions removed
    zero_sample = "Account balance of {$0.00} was 00 past due and 00 balance with 000 remaining."
    cleaned_zero = prepare_nlp_text(zero_sample)
    assert "00" not in cleaned_zero.split()
    assert "000" not in cleaned_zero.split()
    assert "0" not in cleaned_zero.split()
    assert "past due" in cleaned_zero
    assert "balance" in cleaned_zero

    # 3. Meaningful financial amounts, statutory numbers, and terms preserved
    financial_sample = "Under Section 609, credit card fee was 500 dollars for 100 complaints on 12/05/2019."
    cleaned_fin = prepare_nlp_text(financial_sample)
    assert "609" in cleaned_fin
    assert "500" in cleaned_fin
    assert "100" in cleaned_fin
    assert "credit" in cleaned_fin
    assert "card" in cleaned_fin
    assert "fee" in cleaned_fin

    # 4. Words containing letter 'x' preserved (not stripped)
    extra_sample = "I asked for an extra tax explanation."
    cleaned_extra = prepare_nlp_text(extra_sample)
    assert "extra" in cleaned_extra
    assert "tax" in cleaned_extra
    assert "explanation" in cleaned_extra

    # 5. Empty and missing text handled safely
    assert prepare_nlp_text("") == ""
    assert prepare_nlp_text("   ") == ""
    assert prepare_nlp_text(None) == ""


def test_unassigned_for_empty_and_unknown_vocabulary():
    # Fit small model on specific financial terms
    training_texts = pd.Series([
        "Mortgage home loan escrow insurance payment.",
        "Credit card fraud unauthorized charge dispute.",
        "Debt collection agency harassment validation letter.",
    ])
    tfidf, nmf = fit_topic_model(training_texts, n_components=2, random_state=42, min_df=1)

    eval_df = pd.DataFrame({
        "complaint_id": [1, 2, 3, 4],
        "cleaned_narrative": [
            "Mortgage home loan payment.",               # valid text
            "",                                          # empty text
            "   ",                                       # whitespace only
            "xylophone zebra quokka",                    # out-of-vocabulary terms
        ]
    })

    assigned = assign_topics(eval_df, tfidf, nmf, text_col="cleaned_narrative")

    # Valid text gets assigned to one of the candidate topics
    assert assigned.loc[0, "topic_id"].startswith("topic_")
    assert assigned.loc[0, "topic_weight"] > 0.0

    # Empty text receives unassigned
    assert assigned.loc[1, "topic_id"] == "unassigned"
    assert assigned.loc[1, "topic_weight"] == 0.0

    # Whitespace text receives unassigned
    assert assigned.loc[2, "topic_id"] == "unassigned"
    assert assigned.loc[2, "topic_weight"] == 0.0

    # Text with no known TF-IDF vocabulary receives unassigned
    assert assigned.loc[3, "topic_id"] == "unassigned"
    assert assigned.loc[3, "topic_weight"] == 0.0


def test_repeated_identical_text_consistency():
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
            "Debt collector harassed me by telephone at work.",  # identical duplicate
        ]
    })
    assigned = assign_topics(df_eval, tfidf, nmf, text_col="cleaned_narrative")
    assert assigned.loc[0, "topic_id"] == assigned.loc[2, "topic_id"]
    assert assigned.loc[0, "topic_weight"] == assigned.loc[2, "topic_weight"]


def test_topic_counts_reconcile_with_assigned_and_unassigned():
    training_texts = pd.Series([
        "Home mortgage payment late fee escrow.",
        "Credit card interest rate charge balance.",
        "Debt collection agency phone call harassment.",
    ])
    tfidf, nmf = fit_topic_model(training_texts, n_components=2, random_state=42, min_df=1)

    df_eval = pd.DataFrame({
        "complaint_id": range(1, 6),
        "cleaned_narrative": [
            "Home mortgage payment fee.",
            "Credit card interest charge.",
            "",
            "completely unknown terms here",
            "Debt collection agency call.",
        ]
    })
    assigned = assign_topics(df_eval, tfidf, nmf, text_col="cleaned_narrative")

    topic_counts = assigned["topic_id"].value_counts()
    unassigned_count = int(topic_counts.get("unassigned", 0))
    assigned_count = sum(count for tid, count in topic_counts.items() if tid != "unassigned")

    assert assigned_count + unassigned_count == len(df_eval)
    assert unassigned_count == 2
    assert assigned_count == 3


def test_topic_modeling_pipeline_on_sample():
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

    distinct_series = pd.Series(df_sample["cleaned_narrative"].apply(prepare_nlp_text).unique())
    tfidf, nmf = fit_topic_model(distinct_series, n_components=3, random_state=42, min_df=1)

    keywords = extract_topic_keywords(tfidf, nmf, n_words=5)
    assert len(keywords) == 3
    assert all(len(words) == 5 for words in keywords.values())

    df_assigned = assign_topics(df_sample, tfidf, nmf, text_col="cleaned_narrative")

    valid_topic_ids = {"topic_01", "topic_02", "topic_03"}
    assert df_assigned["topic_id"].isin(valid_topic_ids).all()
    assert (df_assigned["topic_weight"] >= 0.0).all()
