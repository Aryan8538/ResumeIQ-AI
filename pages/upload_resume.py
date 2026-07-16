"""
Streamlit Page: Resume Upload and Parse Visualization.
Presents a premium interface for uploading a resume, parsing it,
and visualizing the extracted structured data.
"""

import os
import sys
import json
import logging
import tempfile
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure workspace root is on the path so core can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.parser import parse_resume
    from core.extractor import extract_resume_info
except Exception as e:
    logger.error(f"Failed to import core modules: {e}")
    st.error(f"Failed to import core modules: {e}")

# Inject premium custom CSS and modern fonts
def inject_custom_styles():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        
        <style>
            /* Base Theme overrides */
            html, body, [class*="css"] {
                font-family: 'Outfit', sans-serif;
            }
            
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Outfit', sans-serif;
                font-weight: 600;
                color: #2F3E46;
            }
            
            /* Gradient headers and cards */
            .main-header {
                background: linear-gradient(135deg, #2E5BFF 0%, #8A2CFF 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                font-size: 3rem;
                margin-bottom: 0.5rem;
                text-align: center;
            }
            
            .subtitle {
                font-size: 1.2rem;
                color: #6C757D;
                text-align: center;
                margin-bottom: 2rem;
            }
            
            /* Glassmorphic Container styling */
            .premium-card {
                background: rgba(255, 255, 255, 0.7);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.08);
                margin-bottom: 1.5rem;
            }
            
            /* Custom styled badges for skills */
            .skill-badge {
                display: inline-block;
                background: linear-gradient(135deg, #E2EBFF 0%, #EFE5FF 100%);
                color: #3842B0;
                padding: 0.4rem 0.8rem;
                border-radius: 12px;
                font-size: 0.85rem;
                font-weight: 500;
                margin: 0.25rem;
                border: 1px solid #D2E0FF;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            .skill-badge:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(56, 66, 176, 0.15);
            }
            
            .soft-skill-badge {
                display: inline-block;
                background: linear-gradient(135deg, #E3FAF2 0%, #D1F2E5 100%);
                color: #0F5132;
                padding: 0.4rem 0.8rem;
                border-radius: 12px;
                font-size: 0.85rem;
                font-weight: 500;
                margin: 0.25rem;
                border: 1px solid #C3E6CB;
                transition: transform 0.2s;
            }
            
            .soft-skill-badge:hover {
                transform: translateY(-2px);
            }
            
            .lang-badge {
                display: inline-block;
                background: linear-gradient(135deg, #FFF3CD 0%, #FFEBAA 100%);
                color: #664D03;
                padding: 0.4rem 0.8rem;
                border-radius: 12px;
                font-size: 0.85rem;
                font-weight: 500;
                margin: 0.25rem;
                border: 1px solid #FFE082;
            }
            
            /* Entry lists */
            .entry-card {
                border-left: 4px solid #8A2CFF;
                background-color: #F8F9FA;
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: 0 8px 8px 0;
            }
            
            .entry-title {
                font-weight: 600;
                font-size: 1.1rem;
                color: #2D3748;
            }
            
            .entry-sub {
                font-size: 0.95rem;
                color: #718096;
                margin-bottom: 0.5rem;
            }
            
            .entry-desc {
                font-size: 0.9rem;
                color: #4A5568;
                white-space: pre-line;
            }
            
            /* Contact grid */
            .contact-item {
                display: flex;
                align-items: center;
                font-size: 0.95rem;
                margin-bottom: 0.5rem;
                color: #4A5568;
            }
            
            .contact-item strong {
                width: 100px;
                color: #2D3748;
            }
            
            /* Custom styled Streamlit Expanders */
            .stExpander {
                border: 1px solid #E2E8F0 !important;
                border-radius: 12px !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
                margin-bottom: 0.8rem !important;
                background: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# App UI layout
def main():
    st.set_page_config(
        page_title="ResumeIQ AI - Parser",
        page_icon="📄",
        layout="wide"
    )
    
    inject_custom_styles()
    
    st.markdown('<div class="main-header">ResumeIQ AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Phase 2: Intelligent Document Parsing Engine</div>', unsafe_allow_html=True)
    
    # Left sidebar instructions
    with st.sidebar:
        st.markdown("### About ResumeIQ AI")
        st.markdown(
            "This engine parses structured data from resumes using "
            "rule-based matching, regular expressions, and spaCy NER. "
            "It runs entirely offline and contains no external LLM dependencies."
        )
        st.markdown("---")
        st.markdown("### Supported formats")
        st.markdown("- PDF Resumes (`.pdf`)")
        st.markdown("- Word Resumes (`.docx`)")
        st.markdown("---")
        st.markdown("**Version**: Phase 2 Release")

    # Main uploader
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("### Actions")
        uploaded_file = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx"],
            label_visibility="collapsed"
        )
        
    with col2:
        if uploaded_file is not None:
            st.markdown(f"### Parsing Results: `{uploaded_file.name}`")
            
            # Temporary file writing inside workspace to stay compliant
            temp_dir = os.path.join(os.getcwd(), "data", "temp_resumes")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save uploaded bytes
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            try:
                # Execution
                with st.spinner("Extracting plain text & mapping fields..."):
                    raw_text = parse_resume(temp_file_path)
                    parsed_data = extract_resume_info(raw_text)
                
                st.success("Resume parsed successfully!")
                
                # Visual Layout
                tab_visual, tab_json = st.tabs(["✨ Extracted Profile", "💻 JSON Output"])
                
                with tab_visual:
                    # Row 1: Personal Info and Summary
                    c_info, c_sum = st.columns([2, 3])
                    
                    with c_info:
                        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                        st.subheader("👤 Contact Details")
                        
                        st.markdown(
                            f'<div class="contact-item"><strong>Name:</strong> {parsed_data["name"] or "Not Detected"}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="contact-item"><strong>Email:</strong> {parsed_data["email"] or "Not Detected"}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="contact-item"><strong>Phone:</strong> {parsed_data["phone"] or "Not Detected"}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Links
                        linkedin_val = f'<a href="{parsed_data["linkedin"]}" target="_blank">View Profile</a>' if parsed_data["linkedin"] else "Not Provided"
                        github_val = f'<a href="{parsed_data["github"]}" target="_blank">View Profile</a>' if parsed_data["github"] else "Not Provided"
                        portfolio_val = f'<a href="{parsed_data["portfolio"]}" target="_blank">View Site</a>' if parsed_data["portfolio"] else "Not Provided"
                        
                        st.markdown(
                            f'<div class="contact-item"><strong>LinkedIn:</strong> {linkedin_val}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="contact-item"><strong>GitHub:</strong> {github_val}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="contact-item"><strong>Portfolio:</strong> {portfolio_val}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    with c_sum:
                        st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
                        st.subheader("📝 Professional Summary")
                        summary_text = parsed_data["summary"]
                        if summary_text:
                            st.write(summary_text)
                        else:
                            st.info("No professional summary section was detected in the resume.")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    # Row 2: Skills
                    st.subheader("⚙️ Skills & Competencies")
                    col_t, col_s = st.columns(2)
                    
                    with col_t:
                        st.markdown('<div class="premium-card" style="min-height: 150px;">', unsafe_allow_html=True)
                        st.markdown("#### Technical Skills")
                        if parsed_data["skills"]:
                            badges = "".join(f'<span class="skill-badge">{skill}</span>' for skill in parsed_data["skills"])
                            st.markdown(badges, unsafe_allow_html=True)
                        else:
                            st.write("*No technical skills matched from the database.*")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    with col_s:
                        st.markdown('<div class="premium-card" style="min-height: 150px;">', unsafe_allow_html=True)
                        st.markdown("#### Soft Skills")
                        if parsed_data["soft_skills"]:
                            badges = "".join(f'<span class="soft-skill-badge">{skill}</span>' for skill in parsed_data["soft_skills"])
                            st.markdown(badges, unsafe_allow_html=True)
                        else:
                            st.write("*No soft skills detected.*")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    # Detailed Sections in Expanders
                    with st.expander("🎓 Education", expanded=True):
                        if parsed_data["education"]:
                            for edu in parsed_data["education"]:
                                st.markdown(
                                    f"""
                                    <div class="entry-card">
                                        <div class="entry-title">{edu["degree"]}</div>
                                        <div class="entry-sub">🏫 {edu["institution"]} &nbsp;|&nbsp; 📅 {edu["duration"]}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("No education details structured.")
                            
                    with st.expander("💼 Experience", expanded=True):
                        if parsed_data["experience"]:
                            for job in parsed_data["experience"]:
                                st.markdown(
                                    f"""
                                    <div class="entry-card">
                                        <div class="entry-title">{job["role"]} at {job["company"]}</div>
                                        <div class="entry-sub">📅 {job["duration"]}</div>
                                        <div class="entry-desc">{job["description"]}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("No experience details structured.")
                            
                    with st.expander("🚀 Projects", expanded=False):
                        if parsed_data["projects"]:
                            for proj in parsed_data["projects"]:
                                st.markdown(
                                    f"""
                                    <div class="entry-card">
                                        <div class="entry-title">{proj["title"]}</div>
                                        <div class="entry-sub">📅 {proj["duration"]}</div>
                                        <div class="entry-desc">{proj["description"]}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("No projects structured.")
                            
                    # Languages and Certifications
                    col_lang, col_cert = st.columns(2)
                    with col_lang:
                        with st.expander("🗣️ Languages", expanded=True):
                            if parsed_data["languages"]:
                                badges = "".join(f'<span class="lang-badge">{lang}</span>' for lang in parsed_data["languages"])
                                st.markdown(badges, unsafe_allow_html=True)
                            else:
                                st.info("No languages parsed.")
                                
                    with col_cert:
                        with st.expander("📜 Certifications", expanded=True):
                            if parsed_data["certifications"]:
                                for cert in parsed_data["certifications"]:
                                    st.markdown(f"- {cert}")
                            else:
                                st.info("No certifications parsed.")
                                
                with tab_json:
                    st.subheader("Parsed JSON Payload")
                    st.markdown("Every later phase will consume this object layout directly:")
                    st.json(parsed_data)
                    
            except Exception as e:
                st.error(f"Parsing engine encountered an error: {e}")
                logger.exception("Parsing failure details")
                
            finally:
                # Cleanup temp file
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception as clean_err:
                        logger.warning(f"Failed to delete temp file {temp_file_path}: {clean_err}")
        else:
            st.info("Please upload a PDF or DOCX resume to view extracted information.")

if __name__ == "__main__":
    main()
