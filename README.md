# KURAL — AI-Assisted Financial Complaint & Service Operations Analytics

## Project Overview

Kural (from the Tamil word for "voice") is an analytics project designed to listen to customer voices and extract actionable operational intelligence from consumer financial complaints submitted to the Consumer Financial Protection Bureau (CFPB).

## Business Problem

Financial institutions process thousands of complaints regarding credit reporting, debt collection, mortgage servicing, and banking accounts. Manually reviewing each customer narrative is slow, costly, and difficult to scale. Without automated analytics, organizations risk missing operational bottlenecks, encountering regulatory compliance penalties, and failing to detect systemic service failures.

Kural aims to study:
- Which financial products and operational issues generate the highest complaint volumes.
- How effectively and timely financial institutions respond to consumer grievances.
- The linguistic characteristics and length variations of consumer narratives across product categories.
- How recurring dispute templates and duplicate complaints impact complaint operations.

## Dataset Source

The dataset used in this project is sourced from the historical CFPB complaints extract maintained in the `smltar` repository:
- URL: https://github.com/EmilHvitfeldt/smltar/blob/master/data/complaints.csv.gz
- Format: Gzip-compressed CSV containing 117,214 consumer complaint records from 2019 to early 2020.

## Project Workflow

1. **Data Ingestion & Storage:** Raw compressed complaint records stored in `data/raw/` (ignored by Git for data privacy and repository hygiene).
2. **Data Preparation & Cleaning:** Automated parsing of date fields, standardizing whitespace across categorical columns, lowercasing narrative text while protecting redaction markers, identifying duplicate narratives, and deriving initial temporal and length metrics (`src/prepare_data.py`).
3. **Exploratory Data Analysis:** Systematic inspection of complaint volumes, common issues, monthly timelines, institutional distributions, response categories, and narrative word counts (`notebooks/exploratory_analysis.ipynb`).
4. **Statistical Analysis & Hypothesis Testing:** *(Planned)* Evaluating significant differences in response times and resolution patterns across institutions and product lines.
5. **AI / NLP Modeling:** *(Planned)* Topic extraction, clustering, and text classification on cleaned consumer complaint narratives.
6. **SQL Database Modeling:** *(Planned)* Structured relational tables and analytical queries in `sql/`.
7. **Interactive Dashboard:** *(Planned)* Operational monitoring dashboard in `dashboard/`.

## Current Implemented Work

- **Project Environment & Structure:** Defined repository layout, `.gitignore` rules, and package dependencies in `requirements.txt`.
- **Data Preparation Pipeline:** Implemented `src/prepare_data.py` to clean and standardize 117,214 records, engineer text and temporal features, and output `data/processed/complaints_clean.csv.gz`.
- **Exploratory Data Analysis Notebook:** Created and executed `notebooks/exploratory_analysis.ipynb` covering dataset overview, distribution by product/company/issue, response patterns, and narrative word counts.
- **Analytical Tables:** Exported aggregated CSV tables to `outputs/tables/`:
  - `complaints_by_product.csv`
  - `monthly_complaints.csv`
  - `response_summary.csv`
  - `product_timely_response.csv`
- **Visualizations:** Saved 6 charts to `outputs/figures/`:
  - `complaints_by_product.png`
  - `monthly_complaint_volume.png`
  - `most_frequent_issues.png`
  - `company_responses.png`
  - `timely_response_by_product.png`
  - `narrative_word_count_distribution.png`

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

Open the Jupyter notebook or execute it via command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/exploratory_analysis.ipynb
```

The notebook generates updated figures in `outputs/figures/` and analytical tables in `outputs/tables/`.
