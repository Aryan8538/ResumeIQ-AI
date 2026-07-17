# ResumeIQ AI

ResumeIQ AI is an enterprise-grade, AI-powered resume screening assistant. It enables recruiters to parse resumes, analyze candidates against job descriptions (JDs), identify skill gaps, generate customized interview questions, and converse with a chatbot grounded directly in the candidate's uploaded files using **Retrieval-Augmented Generation (RAG)**.

---

## ✨ Features

- **Multi-Format Parsing**: Extracts text from PDF and DOCX resumes using a robust offline pipeline.
- **AI Summary**: Generates a professional candidate executive summary using `gemini-3.5-flash`.
- **Match Analysis**: Computes a matching percentage, extracts core candidate strengths, analyzes weaknesses, and flags missing skill gaps.
- **Interview Question Generator**: Generates customized Technical, Behavioral (STAR format), and HR interview questions along with ideal response guides.
- **RAG-Powered Chatbot**: Interactive recruiter chatbot grounded in candidate materials with grounding citations (displays matching document segments).

---

## 🛠️ Technology Stack

- **Frontend Dashboard**: Streamlit (Premium layout, custom badges, metric visualizers)
- **Generative AI Core**: Google Gemini 3.5 Flash (`gemini-3.5-flash`)
- **Semantic Vector Embeddings**: Gemini Embeddings (`gemini-embedding-2`)
- **Vector Database**: ChromaDB (Local persistent instance)
- **Offline Parsing**: PyMuPDF, pdfplumber, python-docx, and spaCy NER (`en_core_web_sm`)
- **Dependency Manager**: UV + python-dotenv

---

## 📂 Directory Structure

```text
ResumeIQ-AI/
│
├── app.py                  # Main entry point (Streamlit Dashboard UI)
├── requirements.txt        # PIP dependencies
├── pyproject.toml          # UV project configuration and dependencies
├── .gitignore              # Files to ignore in Git (ignores chroma_db/, .env, etc.)
├── .env.example            # Environment variables template
├── README.md               # Documentation
│
├── services/               # GenAI & core support services
│   ├── parser.py           # Unified document parsing router
│   ├── gemini_service.py   # Gemini API handlers (Summary, Screen, Questions)
│   └── prompts.py          # Unified prompts and structured schemas repository
│
├── rag/                    # Retrieval-Augmented Generation components
│   ├── chunker.py          # Recursive overlapping text-splitting engine
│   ├── embeddings.py       # Computes vector representations via Gemini API
│   ├── vector_store.py     # ChromaDB collection wrapper
│   ├── retriever.py        # Semantic lookup logic
│   └── chatbot.py          # Grounded chat generator and history binder
│
├── core/                   # Offline extraction utilities
│   ├── extractor.py        # Entity and metadata extractor
│   ├── constants.py        # Technical terminology taxomony
│   └── regex_patterns.py   # Regex validation queries
│
└── chroma_db/              # Local persisted vector store (Git ignored)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.13**
- **Gemini API Key**: Obtain one from [Google AI Studio](https://aistudio.google.com/).

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd ResumeIQ-AI
   ```

2. Setup virtual environment and dependencies using `uv` (recommended):
   ```bash
   # Create and activate environment
   uv venv
   .venv\Scripts\activate      # Windows (PowerShell)
   source .venv/bin/activate    # macOS/Linux

   # Install all packages
   uv sync
   ```

3. Setup Environment Variables:
   Copy `.env.example` to `.env` and fill in your Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Add your key inside `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

## 🏃 Running the Application

Start the Streamlit screening assistant:
```bash
streamlit run app.py
```

### Usage Workflow

1. **Upload Documents**: Drag and drop a Candidate Resume (PDF/DOCX) and Job Description (PDF) in the left sidebar.
2. **Process**: Click **"Analyze & Initialize RAG"**.
3. **Explore Dashboard**:
   - **Candidate Profile Tab**: View structured contacts, professional details, and tech/soft skills.
   - **Screen & Matching Tab**: Review the Match Percentage, Recommendation badge, strengths, weaknesses, and categorized interview questions.
   - **AI Recruiter Chat Tab**: Chat with the copilot. Click **"Grounding Sources"** below chatbot answers to audit the exact parts of the resume or JD that backed up the response.
