"""NLP text preparation and topic modeling for Kural financial complaints.

Implements text preprocessing, TF-IDF vectorization, Non-negative Matrix
Factorization (NMF), topic assignment, table generation, and visualizations.
"""

from pathlib import Path
import re
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_nlp_text(text: str) -> str:
    """Prepares consumer complaint text specifically for NLP topic modeling.

    Handles missing text safely, normalizes whitespace, removes CFPB redaction
    placeholders (e.g., 'xxxx', 'xx/xx/xxxx', isolated 'x' tokens), removes
    redacted-amount placeholders and standalone zero-only tokens, and strips
    punctuation while preserving meaningful financial words and numbers.

    Parameters
    ----------
    text : str
        The raw or cleaned narrative string.

    Returns
    -------
    str
        The preprocessed narrative ready for vectorization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text.lower()

    # 1. Remove recognized CFPB amount redactions and amount placeholders
    cleaned = re.sub(r"\{\s*\$?\s*0*(?:\.0*)?\s*\}", " ", cleaned)
    cleaned = re.sub(r"\{\s*\$?\s*x+[\s\.\-x]*\}", " ", cleaned)
    cleaned = re.sub(r"\[\s*(?:amount|amount redacted|\$0\.00)\s*\]", " ", cleaned)

    # Strip decimal cents .00 from currency/numbers (e.g., 500.00 -> 500)
    cleaned = re.sub(r"(?<=\d)\.00(?!\d)", "", cleaned)

    # 2. Remove date-like or slash-separated redaction tokens (e.g. xx/xx/xxxx)
    cleaned = re.sub(r"\b[x]+(?:/[x]+)+\b", " ", cleaned)

    # Remove hyphen-separated redaction tokens (e.g. xxxx-xxxx)
    cleaned = re.sub(r"\b[x]+(?:-[x]+)+\b", " ", cleaned)

    # Remove standalone redaction tokens (2 or more x's, or isolated single x)
    cleaned = re.sub(r"\b[x]{2,}\b", " ", cleaned)
    cleaned = re.sub(r"\b[x]\b", " ", cleaned)

    # 3. Replace punctuation and special characters with space, preserving alphanumeric
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)

    # Remove residual standalone x tokens exposed after punctuation stripping
    cleaned = re.sub(r"\b[x]{2,}\b", " ", cleaned)
    cleaned = re.sub(r"\b[x]\b", " ", cleaned)

    # 4. Remove standalone zero-only tokens (e.g. 0, 00, 000, 0000)
    cleaned = re.sub(r"\b0+\b", " ", cleaned)

    # Normalize whitespace
    return re.sub(r"\s+", " ", cleaned).strip()


def fit_topic_model(
    distinct_texts: pd.Series,
    n_components: int = 16,
    random_state: int = 42,
    max_iter: int = 350,
    min_df: int = 10,
    max_df: float = 0.85,
    max_features: int = 15000,
) -> Tuple[TfidfVectorizer, NMF]:
    """Fits TF-IDF and NMF models on distinct complaint texts.

    Parameters
    ----------
    distinct_texts : pd.Series
        Unique preprocessed complaint texts to avoid duplicate dominance in training.
    n_components : int, default 16
        Number of latent topics to extract.
    random_state : int, default 42
        Seed for reproducibility.
    max_iter : int, default 350
        Maximum iterations for NMF solver.
    min_df : int, default 10
        Minimum document frequency for TF-IDF.
    max_df : float, default 0.85
        Maximum document frequency for TF-IDF.
    max_features : int, default 15000
        Maximum vocabulary size for TF-IDF.

    Returns
    -------
    Tuple[TfidfVectorizer, NMF]
        Fitted vectorizer and NMF model.
    """
    valid_texts = distinct_texts[distinct_texts.str.len() > 0]

    effective_min_df = min(min_df, max(1, len(valid_texts) // 4)) if len(valid_texts) <= min_df else min_df
    effective_max_df = 1.0 if len(valid_texts) < 20 else max_df

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=effective_min_df,
        max_df=effective_max_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(valid_texts)

    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=random_state,
        max_iter=max_iter,
    )
    nmf.fit(tfidf_matrix)

    return tfidf, nmf


def extract_topic_keywords(
    tfidf: TfidfVectorizer, nmf: NMF, n_words: int = 12
) -> Dict[str, List[str]]:
    """Extracts top representative words or phrases for each topic using NMF component weights.

    Parameters
    ----------
    tfidf : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    nmf : NMF
        Fitted NMF model.
    n_words : int, default 12
        Number of representative keywords per topic.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of topic ID (e.g., 'topic_01') to list of top keywords.
    """
    feature_names = tfidf.get_feature_names_out()
    topic_keywords = {}

    for idx, component in enumerate(nmf.components_):
        topic_id = f"topic_{idx + 1:02d}"
        top_indices = component.argsort()[: -n_words - 1 : -1]
        topic_keywords[topic_id] = [feature_names[i] for i in top_indices]

    return topic_keywords


def assign_topics(
    df: pd.DataFrame,
    tfidf: TfidfVectorizer,
    nmf: NMF,
    text_col: str = "cleaned_narrative",
) -> pd.DataFrame:
    """Prepares NLP text and assigns dominant NMF topics to all complaint records.

    Applies the model to distinct prepared texts and maps assignments back to ensure
    identical prepared narratives receive the same topic assignment. Empty texts,
    texts producing no usable TF-IDF vocabulary, and zero-strength NMF results receive
    'unassigned' with weight 0.0.

    Parameters
    ----------
    df : pd.DataFrame
        Complaints dataframe.
    tfidf : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    nmf : NMF
        Fitted NMF model.
    text_col : str, default 'cleaned_narrative'
        Column containing narrative text.

    Returns
    -------
    pd.DataFrame
        Dataframe with added 'topic_id' and 'topic_weight' columns.
    """
    df_result = df.copy()

    nlp_text = df_result[text_col].apply(prepare_nlp_text)

    distinct_series = pd.Series(nlp_text.unique())
    valid_distinct = distinct_series[distinct_series.str.len() > 0]

    topic_map = {}
    weight_map = {}

    if len(valid_distinct) > 0:
        tfidf_distinct = tfidf.transform(valid_distinct)
        weights_distinct = nmf.transform(tfidf_distinct)

        distinct_topic_indices = np.argmax(weights_distinct, axis=1)
        distinct_max_weights = np.max(weights_distinct, axis=1)

        for text_val, t_idx, w_val in zip(valid_distinct, distinct_topic_indices, distinct_max_weights):
            if w_val <= 0.0 or np.isclose(w_val, 0.0):
                topic_map[text_val] = "unassigned"
                weight_map[text_val] = 0.0
            else:
                topic_map[text_val] = f"topic_{t_idx + 1:02d}"
                weight_map[text_val] = float(round(w_val, 6))

    # Empty text fallback maps to unassigned
    topic_map[""] = "unassigned"
    weight_map[""] = 0.0

    df_result["topic_id"] = nlp_text.map(topic_map).fillna("unassigned")
    df_result["topic_weight"] = nlp_text.map(weight_map).fillna(0.0)

    return df_result


def _sort_topic_ids(topic_ids: List[str]) -> List[str]:
    """Sort helper ensuring topic_01..topic_16 appear in order followed by unassigned."""
    def sort_key(tid: str):
        if tid.startswith("topic_"):
            try:
                return (0, int(tid.split("_")[1]))
            except ValueError:
                return (0, 999)
        return (1, tid)

    return sorted(topic_ids, key=sort_key)


def generate_topic_summary_tables(
    df_with_topics: pd.DataFrame,
    tfidf: TfidfVectorizer,
    nmf: NMF,
    tables_dir: Path,
) -> None:
    """Generates and exports the five required aggregated topic summary tables.

    Parameters
    ----------
    df_with_topics : pd.DataFrame
        Complaints dataframe containing 'topic_id'.
    tfidf : TfidfVectorizer
        Fitted vectorizer.
    nmf : NMF
        Fitted NMF model.
    tables_dir : Path
        Directory to save CSV tables.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    total_assigned = len(df_with_topics)

    # A. Topic Keywords (for the 16 candidate NMF topics)
    keywords_dict = extract_topic_keywords(tfidf, nmf, n_words=12)
    topic_counts = df_with_topics["topic_id"].value_counts()

    keywords_rows = []
    for topic_id, words in keywords_dict.items():
        keywords_rows.append({
            "topic_id": topic_id,
            "top_keywords": ", ".join(words),
            "complaint_count": int(topic_counts.get(topic_id, 0)),
        })
    df_keywords = pd.DataFrame(keywords_rows).sort_values("topic_id").reset_index(drop=True)
    df_keywords.to_csv(tables_dir / "topic_keywords.csv", index=False)

    # B. Topic Distribution (accounts for every complaint, including unassigned)
    dist_counts = df_with_topics["topic_id"].value_counts()
    sorted_topics = _sort_topic_ids(list(dist_counts.index))
    df_dist = pd.DataFrame({
        "topic_id": sorted_topics,
        "complaint_count": [int(dist_counts[t]) for t in sorted_topics],
    })
    df_dist["percentage"] = (df_dist["complaint_count"] / total_assigned * 100).round(2)
    df_dist.to_csv(tables_dir / "topic_distribution.csv", index=False)

    # C. Topics Across Financial Products
    prod_totals = df_with_topics.groupby("product")["complaint_id"].count().rename("product_total")
    df_prod = df_with_topics.groupby(["product", "topic_id"]).size().reset_index(name="complaint_count")
    df_prod = df_prod.merge(prod_totals, on="product")
    df_prod["percentage"] = (df_prod["complaint_count"] / df_prod["product_total"] * 100).round(2)
    df_prod = df_prod[["product", "topic_id", "complaint_count", "percentage"]]
    df_prod = df_prod.sort_values(["product", "complaint_count"], ascending=[True, False]).reset_index(drop=True)
    df_prod.to_csv(tables_dir / "topic_by_product.csv", index=False)

    # D. Topics and Recorded Relief
    df_relief = df_with_topics.groupby("topic_id").agg(
        total_complaints=("has_recorded_relief", "count"),
        complaints_with_recorded_relief=("has_recorded_relief", "sum"),
    ).reset_index()
    df_relief["recorded_relief_percentage"] = (
        df_relief["complaints_with_recorded_relief"] / df_relief["total_complaints"] * 100
    ).round(2)
    df_relief["sort_rank"] = df_relief["topic_id"].apply(
        lambda x: (0, int(x.split("_")[1])) if x.startswith("topic_") else (1, x)
    )
    df_relief = df_relief.sort_values("sort_rank").drop(columns=["sort_rank"]).reset_index(drop=True)
    df_relief.to_csv(tables_dir / "topic_recorded_relief.csv", index=False)

    # E. Topic Issue Alignment (top 3 source issues per topic)
    issue_rows = []
    for topic_id in sorted_topics:
        group = df_with_topics[df_with_topics["topic_id"] == topic_id]
        topic_total = len(group)
        top_issues = group["issue"].value_counts().head(3)
        for issue_name, count in top_issues.items():
            pct = round((count / topic_total) * 100, 2)
            issue_rows.append({
                "topic_id": topic_id,
                "issue": issue_name,
                "complaint_count": int(count),
                "percentage_within_topic": pct,
            })
    df_issue_align = pd.DataFrame(issue_rows)
    df_issue_align.to_csv(tables_dir / "topic_issue_alignment.csv", index=False)


def generate_topic_visualizations(df_with_topics: pd.DataFrame, figures_dir: Path) -> None:
    """Generates and exports the two required topic modeling figures.

    Parameters
    ----------
    df_with_topics : pd.DataFrame
        Complaints dataframe containing 'topic_id'.
    figures_dir : Path
        Directory to save figures.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    total_assigned = len(df_with_topics)

    # 1. Topic Distribution Bar Chart
    dist_counts = df_with_topics["topic_id"].value_counts()
    sorted_topics = _sort_topic_ids(list(dist_counts.index))
    # Invert so topic_01 is at the top of the horizontal bar chart
    plot_topics = list(reversed(sorted_topics))
    plot_counts = [dist_counts[t] for t in plot_topics]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(plot_topics, plot_counts, color="#1f77b4", edgecolor="none")
    ax.set_title("Complaint Count by Candidate Topic", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Number of Complaints", fontsize=11)
    ax.set_ylabel("Candidate Topic ID", fontsize=11)
    ax.xaxis.set_major_formatter("{x:,.0f}")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        pct = w / total_assigned * 100
        ax.text(w + 300, bar.get_y() + bar.get_height() / 2, f"{w:,.0f} ({pct:.1f}%)", va="center", fontsize=9)

    ax.set_xlim(0, max(plot_counts) * 1.18)
    plt.tight_layout()
    plt.savefig(figures_dir / "topic_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Topic-Product Heatmap (percentages within each product category)
    pivot_pct = pd.crosstab(
        df_with_topics["product"],
        df_with_topics["topic_id"],
        normalize="index"
    ) * 100

    heatmap_cols = _sort_topic_ids([c for c in pivot_pct.columns])
    pivot_pct = pivot_pct.reindex(columns=heatmap_cols)

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(pivot_pct.values, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(np.arange(pivot_pct.shape[1]))
    ax.set_yticks(np.arange(pivot_pct.shape[0]))
    ax.set_xticklabels(pivot_pct.columns, rotation=45, ha="right", fontsize=9.5)
    ax.set_yticklabels(pivot_pct.index, fontsize=9.5)

    ax.set_title("Topic Prevalence Across Financial Products (% of Complaints Within Product)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Candidate Topic ID", fontsize=11)
    ax.set_ylabel("Product Category", fontsize=11)

    for i in range(pivot_pct.shape[0]):
        for j in range(pivot_pct.shape[1]):
            val = pivot_pct.iloc[i, j]
            color = "white" if val > 25 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% Within Product Category", fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "topic_product_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_text_analysis(
    input_path: str = "data/processed/complaints_clean.csv.gz",
    output_path: str = "data/processed/complaints_with_topics.csv.gz",
) -> pd.DataFrame:
    """Executes the full NLP topic modeling pipeline and saves outputs.

    Parameters
    ----------
    input_path : str
        Path to prepared complaints dataset.
    output_path : str
        Path to save dataset with assigned topics.

    Returns
    -------
    pd.DataFrame
        Complaints dataframe with topic assignments.
    """
    raw_input = Path(input_path)
    output_file = Path(output_path)
    figures_dir = Path("outputs/figures")
    tables_dir = Path("outputs/tables")

    if not raw_input.exists():
        raise FileNotFoundError(f"Prepared complaints file not found: {raw_input}")

    print("Loading prepared complaints data...")
    df = pd.read_csv(raw_input)
    print(f"Loaded {len(df):,} complaints.")

    print("Preparing distinct NLP narrative texts for training...")
    nlp_series = df["cleaned_narrative"].apply(prepare_nlp_text)
    distinct_texts = pd.Series(nlp_series.unique())
    print(f"Fitting on {len(distinct_texts):,} distinct prepared texts...")

    tfidf, nmf = fit_topic_model(distinct_texts, n_components=16, random_state=42, max_iter=350)
    print("TF-IDF and NMF model fitted successfully.")

    print("Assigning topics to all complaints...")
    df_with_topics = assign_topics(df, tfidf, nmf, text_col="cleaned_narrative")

    unassigned_count = int((df_with_topics["topic_id"] == "unassigned").sum())
    assigned_count = len(df_with_topics) - unassigned_count
    print(f"Topic assignment summary: {assigned_count:,} assigned, {unassigned_count:,} unassigned.")

    print(f"Saving complaints with topic assignments to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_with_topics.to_csv(output_file, index=False, compression="gzip")
    print("Dataset saved successfully.")

    print("Generating summary tables...")
    generate_topic_summary_tables(df_with_topics, tfidf, nmf, tables_dir)

    print("Generating topic visualizations...")
    generate_topic_visualizations(df_with_topics, figures_dir)

    print("Topic modeling pipeline completed successfully.")
    return df_with_topics


if __name__ == "__main__":
    run_text_analysis()
