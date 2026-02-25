# Finance RAG System

A specialized Retrieval-Augmented Generation (RAG) system for financial documents (10-K, 10-Q, 8-K), designed to handle complex tables, charts, and unstructured text with high precision.

## 🚀 Features

- **Specialized Parsing**: Handles PDF tables and charts using vision models.
- **Semantic Chunking**: Context-aware chunking for financial narratives.
- **Dual-Database Retrieval (HyPE)**: Uses Hypothetical Document Embeddings to improve retrieval accuracy.
- **Verification Loop**: Self-correcting retrieval that verifies answers against source text.
- **Batch Processing**: Efficiently ingest entire directories of SEC filings.

---

## 🛠️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/finance-rag.git
    cd finance-rag
    ```

2.  **Install `uv` (if not already installed):**
    (MacOS)

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    (Windows)

    ```bash
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

3.  **Install backend dependencies:**

    ```bash
    uv sync
    ```

4.  **Install frontend dependencies:**

    ```bash
    cd frontend && npm install && cd ..
    ```

5.  **Configure `.env`:**
    Create a `.env` file with your API keys:

    ```bash
    OPENAI_API_KEY=your_key_here
    QDRANT_API_KEY=your_key_here
    SEC_API_KEY=your_sec_api_key_here  # For data collection
    ```

6.  **Update `config.yaml`:**
    Adjust model settings, vector store paths, and retrieval parameters in `config.yaml` as needed.

---

## 📊 Data Collection

The system includes a built-in scraper to fetch financial documents (10-K, 10-Q, 8-K) directly from the SEC EDGAR database using the [SEC API](https://sec-api.io/).

### 1. Prerequisites

You need an API key from [Overview - SEC API](https://sec-api.io/). Ensure it is added to your `.env` file as shown above.

### 2. Running the Scraper

The scraper is located at `src/data_collection/sec_scrapper.py`. It is configured to download filings for a pre-defined list of companies (mostly S&P 500 top constituents) for the years 2023-2026.

To run the scraper:

```bash
python -m src.data_collection.sec_scrapper
```

You can modify `src/data_collection/sec_scrapper.py` to adjust the `COMPANIES` list or the date range.

### 3. Output

Files will be saved in the `data/pdf` directory with the following structure:

```
data/pdf/
├── AAPL/
│   ├── 10-K/
│   │   └── AAPL_10-K_0000320193-23-000106.pdf
│   └── 10-Q/
│       └── ...
├── MSFT/
│   └── ...
```

> **Note:** The `ingest.py` script (described below) scans a single directory. To ingest these files, you can either:
>
> - Point the ingestion script to specific sub-folders (e.g., `python ingest.py --directory data/pdf/AAPL/10-K/`)
> - Move/Copy the PDF files you want to process into a single flat directory (e.g., `data/filings/`).

---

## 📥 Data Ingestion

The system supports two main ingestion modes: **Batch (Recommended)** and **Manual**.

### 1. Batch Ingestion (Auto-Metadata)

This is the easiest way to ingest multiple documents. Simply rename your files and point the script to the directory.

#### **Filename Format Requirements**

Files **MUST** follow this naming convention for automatic metadata extraction:

```
COMPANY_DOCTYPE_ACCESSION.pdf
```

- **COMPANY**: Ticker symbol (Uppercase, e.g., `AAPL`, `MSFT`)
- **DOCTYPE**: Document type (e.g., `10-K`, `10-Q`)
- **ACCESSION**: SEC Accession Number (Format: `XXXXXXXXXX-YY-ZZZZZZ`). The year `YY` is used as the filing date year.

#### **Examples:**

- ✅ `AAPL_10-K_0000320193-25-000079.pdf` (Year 2025)
- ✅ `MSFT_10-Q_0001564590-24-000123.pdf` (Year 2024)
- ❌ `apple_report.pdf` (Invalid)

#### **Run Batch Ingestion:**

```bash
python ingest.py --directory data/filings/
```

The system will:

1.  Scan the directory.
2.  Extract metadata from filenames.
3.  Parse, chunk, and index all valid files.

---

### 2. Manual Ingestion

Use this for single files or when you want to explicitly specify metadata.

**Single File:**

```bash
python ingest.py --file data/my_doc.pdf --company AAPL --doc-type 10-K --date 2023-12-31
```

**Batch with Same Metadata (Not Recommended):**
Forces the same metadata for ALL files in the directory.

```bash
python ingest.py --directory data/filings/ --company AAPL --doc-type 10-K --date 2023-12-31
```

---

## 🔎 Querying

Once documents are ingested, you can query them using `query.py`.

### Interactive Mode

Start a chat session with your financial data:

```bash
python query.py
```

- Type your question (e.g., _"What was Apple's revenue in 2023?"_)
- Type `quit` to exit.

### Single Query

Run a quick one-off question:

```bash
python query.py --query "What are the key risk factors for Microsoft?"
```

### Filtered Query

Narrow down sources by company, document type, or date:

```bash
python query.py --query "Revenue growth?" --company AAPL --doc-type 10-K
```

### Batch Query Mode

Process a list of questions from a text file (one question per line):

```bash
python query.py --batch sample_queries.txt > results.txt
```

---

## ⚙️ Advanced Options

- **Skip HyPE**: To speed up ingestion (skips generating hypothetical questions), add `--skip-hype`:
  ```bash
  python ingest.py --directory data/filings/ --skip-hype
  ```

---

## 📂 Project Structure

- `ingest.py`: Main script for document ingestion.
- `query.py`: Main script for querying.
- `config.yaml`: Configuration for models, vector stores, and chunking.
- `src/`: Source code for parsing, chunking, embedding, and generation.
