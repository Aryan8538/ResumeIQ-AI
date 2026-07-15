# ResumeIQ-AI

An AI-powered resume analysis, matching, and optimization tool. ResumeIQ-AI uses LLMs and NLP techniques to parse resumes, match them against job descriptions, evaluate skills, and generate feedback reports.

## Directory Structure

```text
ResumeIQ-AI/
│
├── app.py                  # Main entry point (Streamlit app)
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── .gitignore              # Files to ignore in Git
├── .env.example            # Environment variables template
│
├── assets/                 # Images, logos, static UI assets
│
├── data/                   # Data storage
│   ├── raw/                # Uploaded/unprocessed resumes and job descriptions
│   ├── processed/          # Parsed/structured text and metadata
│   └── external/           # Reference datasets (e.g., skill taxonomies)
│
├── models/                 # Local machine learning models or embeddings
│
├── training/               # Model training scripts and pipelines
│
├── core/                   # Core business logic (resume parsing, grading)
│
├── llm/                    # LLM integration layers, wrappers, and prompts
│
├── pages/                  # Streamlit multi-page dashboard views
│
├── reports/                # Generated analysis and feedback reports
│
├── utils/                  # Helper utilities (file IO, system helpers)
│
├── exports/                # Exported CSVs, PDFs, and JSON files
│
├── docs/                   # Additional documentation, design specs, user guides
│
└── tests/                  # Unit and integration tests
```

## Getting Started

### Prerequisites
- Python 3.9+
- Pip (Python Package Installer)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ResumeIQ-AI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Copy `.env.example` to `.env` and fill in your API keys and configuration details.
   ```bash
   cp .env.example .env
   ```

### Running the App

Start the Streamlit dashboard:
```bash
streamlit run app.py
```
