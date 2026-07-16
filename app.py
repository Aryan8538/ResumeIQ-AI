import streamlit as st

st.set_page_config(
    page_title="ResumeIQ AI - Home",
    page_icon="📄",
    layout="wide"
)

# Custom premium styling
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        .title-text {
            background: linear-gradient(135deg, #2E5BFF 0%, #8A2CFF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3.5rem;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        .tagline {
            font-size: 1.4rem;
            color: #6C757D;
            text-align: center;
            margin-bottom: 3rem;
        }
        .card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
            margin-bottom: 2rem;
        }
        .card h3 {
            margin-top: 0;
            color: #2E5BFF;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title-text">ResumeIQ AI</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Enterprise-Grade Intelligent Resume Parsing Dashboard</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="card">
            <h3>🚀 Welcome to Phase 2</h3>
            <p>ResumeIQ AI has been updated with its Phase 2 modular Parsing Engine. This release is capable of processing resumes in <strong>PDF</strong> and <strong>DOCX</strong> formats and extracting structured fields.</p>
            <p>To try the parser, select <strong>upload_resume</strong> from the left sidebar navigation menu.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <h3>🛠️ Engine Architecture</h3>
            <p>Our parsing pipeline operates fully offline with complete separation of concerns:</p>
            <ul>
                <li><strong>pdf_parser & docx_parser</strong>: Raw text extraction.</li>
                <li><strong>regex_patterns</strong>: Consolidated search queries.</li>
                <li><strong>skills</strong>: Technical taxonomy databases.</li>
                <li><strong>extractor</strong>: Heuristic extraction & spaCy NLP mapping.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )