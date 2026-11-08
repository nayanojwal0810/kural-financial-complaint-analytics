# KURAL — AI-Assisted Financial Complaint & Service Operations Analytics

## Project Overview

Kural (from the Tamil word for "voice") is an analytics project designed to listen to customer voices and extract actionable operational intelligence from consumer financial complaints submitted to the Consumer Financial Protection Bureau (CFPB).

## Business Problem

Financial institutions receive large volumes of consumer complaints across products such as credit reporting, debt collection, mortgages, and consumer banking. Manually reading through thousands of complaint narratives is time-consuming and difficult to do consistently. Without structured complaint analytics, organizations struggle to identify recurring issues, track response timeliness, and understand patterns in consumer grievances.

Kural aims to study:
- Which financial products and issues appear most frequently in consumer complaints.
- How recorded company responses and response timeliness vary across product categories.
- The distribution of narrative lengths and characteristics across different complaint types.
- The prevalence of duplicate narratives in historical complaint data.

## Dataset Source

The dataset used in this project is sourced from the historical CFPB complaints extract maintained in the `smltar` repository:
- URL: https://github.com/EmilHvitfeldt/smltar/blob/master/data/complaints.csv.gz
- Format: Gzip-compressed CSV containing 117,214 consumer complaint records from 2019 to early 2020.

## Project Workflow

1. **Data Ingestion & Storage:** Raw compressed complaint records stored in `data/raw/` (ignored by Git for data privacy and repository hygiene).
2. **Data Preparation & Cleaning:** Automated parsing of date fields, standardizing whitespace across categorical columns, lowercasing narrative text while protecting redaction markers, identifying duplicate narratives, and deriving initial temporal and length metrics (`src/prepare_data.py`).
3. **Exploratory Data Analysis:** Systematic inspection of complaint volumes, common issues, monthly timelines, institutional distributions, response categories, and narrative word counts (`notebooks/exploratory_analysis.ipynb`).
4. **Statistical Analysis & Hypothesis Testing:** Evaluating differences in response timeliness, recorded relief categories, and narrative lengths across product categories (`notebooks/statistical_analysis.ipynb`).
5. **AI / NLP Modeling:** *(Planned)* Topic extraction, clustering, and text classification on cleaned consumer complaint narratives.
6. **SQL Database Modeling:** *(Planned)* Structured relational tables and analytical queries in `sql/`.
7. **Interactive Dashboard:** *(Planned)* Operational monitoring dashboard in `dashboard/`.

## Current Implemented Work

- **Project Environment & Structure:** Defined repository layout, `.gitignore` rules, and package dependencies in `requirements.txt`.
- **Data Preparation Pipeline:** Implemented `src/prepare_data.py` to clean and standardize 117,214 records, engineer text and temporal features, and output `data/processed/complaints_clean.csv.gz`.
- **Exploratory Data Analysis Notebook:** Created and executed `notebooks/exploratory_analysis.ipynb` covering dataset overview, distribution by product/company/issue, response patterns, and narrative word counts.
- **Statistical Analysis Notebook:** Created and executed `notebooks/statistical_analysis.ipynb` evaluating response timeliness, recorded relief proportions, and narrative word counts across product categories.
- **Analytical Tables:** Exported aggregated CSV tables to `outputs/tables/`:
  - `complaints_by_product.csv`
  - `monthly_complaints.csv`
  - `response_summary.csv`
  - `product_timely_response.csv`
  - `product_timeliness_statistics.csv`
  - `product_recorded_relief.csv`
  - `narrative_length_by_product.csv`
- **Visualizations:** Saved analytical charts to `outputs/figures/`:
  - `complaints_by_product.png`
  - `monthly_complaint_volume.png`
  - `most_frequent_issues.png`
  - `company_responses.png`
  - `timely_response_by_product.png`
  - `narrative_word_count_distribution.png`
  - `product_timeliness_intervals.png`
  - `recorded_relief_by_product.png`
  - `narrative_length_by_product.png`

## Setup and Execution

### 1. Requirements

Install required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Place `complaints.csv.gz` inside `data/raw/` and execute the preparation script:

```bash
python src/prepare_data.py
```

This creates the cleaned dataset at `data/processed/complaints_clean.csv.gz`.

### 3. Exploratory Analysis

Execute the exploratory data analysis notebook:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/exploratory_analysis.ipynb
```

### 4. Statistical Analysis

Execute the statistical analysis notebook:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/statistical_analysis.ipynb
```
