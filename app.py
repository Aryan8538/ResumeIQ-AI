"""
Main Entrypoint: ResumeIQ-AI 2.0 Streamlit Dashboard.
Implements the sidebar file handlers, candidate analysis parser, and
the RAG recruiter chatbot grounded in candidate materials.
"""
import os
import sys
import logging
import streamlit as st

# Ensure workspace root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.parser import parse_resume_file, parse_jd_file
from core.extractor import extract_resume_info
from services.gemini_service import (
    generate_resume_summary,
    analyze_resume_vs_jd,
    generate_interview_questions
)
from rag.chunker import chunk_text
from rag.vector_store import reset_db, index_document_chunks
from rag.chatbot import generate_rag_response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Premium style injections
def inject_custom_styles():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        
        <style>
            /* Global font override */
            html, body, [class*="css"] {
                font-family: 'Outfit', sans-serif;
            }
            
            /* Gradient main headers */
            .main-header {
                background: linear-gradient(135deg, #2E5BFF 0%, #8A2CFF 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                font-size: 3.2rem;
                margin-bottom: 0.2rem;
                text-align: center;
            }
            
            .subtitle {
                font-size: 1.25rem;
                color: #6C757D;
                text-align: center;
                margin-bottom: 2.5rem;
            }
            
            /* Modern Card Container */
            .premium-card {
                background: rgba(255, 255, 255, 0.7);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.06);
                margin-bottom: 1.5rem;
            }
            
            /* Match Gauge Panel */
            .match-gauge-panel {
                background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 2rem;
                text-align: center;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
            }
            
            .match-score-text {
                font-size: 4rem;
                font-weight: 800;
                color: #2E5BFF;
                margin: 0;
                line-height: 1;
            }
            
            .match-label {
                font-size: 1.1rem;
                color: #475569;
                font-weight: 500;
                margin-top: 0.5rem;
            }
            
            /* Custom Badges */
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
            }
            
            .missing-skill-badge {
                display: inline-block;
                background: linear-gradient(135deg, #FFF5F5 0%, #FFE3E3 100%);
                color: #C53030;
                padding: 0.4rem 0.8rem;
                border-radius: 12px;
                font-size: 0.85rem;
                font-weight: 500;
                margin: 0.25rem;
                border: 1px solid #FEB2B2;
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
            
            .recommendation-badge {
                display: inline-block;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: 700;
                font-size: 1rem;
                margin-top: 0.5rem;
            }
            
            .rec-strong {
                background-color: #DEF7EC;
                color: #03543F;
                border: 1px solid #84E1BC;
            }
            
            .rec-moderate {
                background-color: #FEF3C7;
                color: #92400E;
                border: 1px solid #FCD34D;
            }
            
            .rec-unsuitable {
                background-color: #FDE8E8;
                color: #9B1C1C;
                border: 1px solid #F8B4B4;
            }
            
            /* Profile details */
            .profile-item {
                font-size: 0.95rem;
                margin-bottom: 0.6rem;
                color: #4A5568;
            }
            
            .profile-item strong {
                width: 100px;
                display: inline-block;
                color: #2D3748;
            }
            
            .section-card {
                border-left: 4px solid #8A2CFF;
                background-color: #F8F9FA;
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: 0 8px 8px 0;
            }
            
            .section-title {
                font-weight: 600;
                font-size: 1.1rem;
                color: #2D3748;
            }
            
            .section-sub {
                font-size: 0.95rem;
                color: #718096;
                margin-bottom: 0.4rem;
            }
            
            .section-desc {
                font-size: 0.9rem;
                color: #4A5568;
                white-space: pre-line;
            }
            
            /* Sources display */
            .citation-container {
                font-size: 0.85rem;
                background-color: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 0.8rem;
                margin-top: 0.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(
        page_title="ResumeIQ AI - Screening Copilot",
        page_icon="📄",
        layout="wide"
    )
    
    inject_custom_styles()
    
    # Header Area
    st.markdown('<div class="main-header">ResumeIQ AI 2.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Candidate Screening & RAG Chatbot</div>', unsafe_allow_html=True)
    
    # Initialize session state variables
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""
    if "parsed_data" not in st.session_state:
        st.session_state.parsed_data = {}
    if "ai_summary" not in st.session_state:
        st.session_state.ai_summary = ""
    if "analysis" not in st.session_state:
        st.session_state.analysis = {}
    if "questions" not in st.session_state:
        st.session_state.questions = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar Panel
    with st.sidebar:
        st.header("📂 Document Uploads")
        st.write("Provide the candidate's resume and target job requirements to start the analysis.")
        
        resume_file = st.file_uploader(
            "Upload Candidate Resume (PDF/DOCX)",
            type=["pdf", "docx"],
            help="Supported file formats: PDF, DOCX"
        )
        
        jd_file = st.file_uploader(
            "Upload Job Description (PDF)",
            type=["pdf"],
            help="Supported file formats: PDF"
        )
        
        st.markdown("---")
        
        # Action button
        analyze_clicked = st.button("🚀 Analyze & Initialize RAG", use_container_width=True)
        
        if analyze_clicked:
            if not resume_file:
                st.error("Please upload a candidate resume.")
            elif not jd_file:
                st.error("Please upload a target job description.")
            else:
                # Create upload directory inside project
                upload_dir = os.path.join(os.getcwd(), "data", "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                
                resume_path = os.path.join(upload_dir, resume_file.name)
                jd_path = os.path.join(upload_dir, jd_file.name)
                
                # Save uploaded bytes
                with open(resume_path, "wb") as f:
                    f.write(resume_file.getbuffer())
                with open(jd_path, "wb") as f:
                    f.write(jd_file.getbuffer())
                
                # Processing pipeline
                try:
                    with st.spinner("Extracting plain text from files..."):
                        resume_text = parse_resume_file(resume_path)
                        jd_text = parse_jd_file(jd_path)
                        
                        st.session_state.resume_text = resume_text
                        st.session_state.jd_text = jd_text
                        
                    with st.spinner("Extracting offline profiles & entities..."):
                        # Offline regex & NLP parser
                        parsed_profile = extract_resume_info(resume_text)
                        st.session_state.parsed_data = parsed_profile
                        
                    with st.spinner("Calling Gemini 3.5 Flash for analysis..."):
                        # AI summary
                        summary = generate_resume_summary(resume_text)
                        st.session_state.ai_summary = summary
                        
                        # Compare JD and Resume
                        analysis = analyze_resume_vs_jd(resume_text, jd_text)
                        st.session_state.analysis = analysis
                        
                        # Generate Interview Questions
                        questions = generate_interview_questions(resume_text, jd_text)
                        st.session_state.questions = questions
                        
                    with st.spinner("Creating text vector embeddings & building RAG database..."):
                        # Segment into chunks
                        resume_chunks = chunk_text(resume_text, chunk_size=500, chunk_overlap=100)
                        jd_chunks = chunk_text(jd_text, chunk_size=500, chunk_overlap=100)
                        
                        # Initialize vector database
                        reset_db()
                        index_document_chunks(resume_chunks, "resume")
                        index_document_chunks(jd_chunks, "job_description")
                        
                    st.session_state.processed = True
                    st.session_state.chat_history = []  # Clear previous chat
                    st.success("Analysis complete! Review the profile and matching results.")
                    
                except Exception as ex:
                    st.error(f"Pipeline error: {ex}")
                    logger.exception("Screening execution failure")
                    
                finally:
                    # Clean up temp files
                    for path in [resume_path, jd_path]:
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                            except Exception as cleanup_err:
                                logger.warning(f"Could not remove temp file {path}: {cleanup_err}")
                                
        st.markdown("---")
        st.markdown("**Version**: ResumeIQ-AI 2.0")
        st.markdown("**Model**: Gemini 3.5 Flash")

    # Main dashboard view
    if not st.session_state.processed:
        # Welcome Page
        st.markdown(
            """
            <div class="premium-card">
                <h3 style="color:#2E5BFF; margin-top:0;">👋 Welcome to ResumeIQ-AI 2.0</h3>
                <p>An enterprise-grade recruiter screening assistant powered by <strong>Generative AI</strong> and <strong>Retrieval-Augmented Generation (RAG)</strong>.</p>
                <p>Upload a Candidate Resume and a Job Description in the sidebar to:</p>
                <ul>
                    <li>Parse structured profile details and skill sets.</li>
                    <li>Generate professional summaries and comparative analyses.</li>
                    <li>Evaluate matching percentages, weaknesses, and missing skill badges.</li>
                    <li>Generate customized technical, behavioral, and cultural interview questions.</li>
                    <li>Query an interactive chatbot grounded directly in the candidate's files.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Create Main Area Tabs
        tab_profile, tab_match, tab_chat = st.tabs([
            "✨ Candidate Profile", 
            "📊 Screen & Matching", 
            "💬 AI Recruiter Chat"
        ])
        
        # =========================================================================
        # TAB 1: CANDIDATE PROFILE
        # =========================================================================
        with tab_profile:
            parsed = st.session_state.parsed_data
            
            c_info, c_sum = st.columns([2, 3])
            
            with c_info:
                st.markdown('<div class="premium-card" style="height:100%;">', unsafe_allow_html=True)
                st.subheader("👤 Contact Information")
                st.markdown(f'<div class="profile-item"><strong>Name:</strong> {parsed.get("name") or "Not Detected"}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="profile-item"><strong>Email:</strong> {parsed.get("email") or "Not Detected"}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="profile-item"><strong>Phone:</strong> {parsed.get("phone") or "Not Detected"}</div>', unsafe_allow_html=True)
                
                # Links
                ln = parsed.get("linkedin")
                gh = parsed.get("github")
                pf = parsed.get("portfolio")
                ln_link = f'<a href="{ln}" target="_blank">View Profile</a>' if ln else "Not Provided"
                gh_link = f'<a href="{gh}" target="_blank">View Profile</a>' if gh else "Not Provided"
                pf_link = f'<a href="{pf}" target="_blank">View Portfolio</a>' if pf else "Not Provided"
                
                st.markdown(f'<div class="profile-item"><strong>LinkedIn:</strong> {ln_link}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="profile-item"><strong>GitHub:</strong> {gh_link}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="profile-item"><strong>Portfolio:</strong> {pf_link}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c_sum:
                st.markdown('<div class="premium-card" style="height:100%;">', unsafe_allow_html=True)
                st.subheader("📝 Professional Summary (Gemini Generated)")
                st.write(st.session_state.ai_summary)
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Skills row
            st.markdown("### ⚙️ Skills & Competencies")
            c_ts, c_ss = st.columns(2)
            
            with c_ts:
                st.markdown('<div class="premium-card" style="min-height:160px;">', unsafe_allow_html=True)
                st.markdown("#### Technical Skills")
                tech_skills = parsed.get("skills", [])
                if tech_skills:
                    badges = "".join(f'<span class="skill-badge">{s}</span>' for s in tech_skills)
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.info("No technical skills detected by parser database.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c_ss:
                st.markdown('<div class="premium-card" style="min-height:160px;">', unsafe_allow_html=True)
                st.markdown("#### Soft Skills")
                soft_skills = parsed.get("soft_skills", [])
                if soft_skills:
                    badges = "".join(f'<span class="soft-skill-badge">{s}</span>' for s in soft_skills)
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.info("No soft skills parsed.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Work experience details
            with st.expander("💼 Professional Experience", expanded=True):
                exp_list = parsed.get("experience", [])
                if exp_list:
                    for job in exp_list:
                        st.markdown(
                            f"""
                            <div class="section-card">
                                <div class="section-title">{job.get("role")} at {job.get("company")}</div>
                                <div class="section-sub">📅 {job.get("duration") or "Not Stated"}</div>
                                <div class="section-desc">{job.get("description") or ""}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No experience details structures found.")
                    
            # Education details
            with st.expander("🎓 Education & Credentials", expanded=True):
                edu_list = parsed.get("education", [])
                if edu_list:
                    for edu in edu_list:
                        st.markdown(
                            f"""
                            <div class="section-card">
                                <div class="section-title">{edu.get("degree")}</div>
                                <div class="section-sub">🏫 {edu.get("institution")} &nbsp;|&nbsp; 📅 {edu.get("duration") or ""}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No education entries structured.")
                    
            # Projects details
            with st.expander("🚀 Projects & Contributions", expanded=False):
                proj_list = parsed.get("projects", [])
                if proj_list:
                    for proj in proj_list:
                        st.markdown(
                            f"""
                            <div class="section-card">
                                <div class="section-title">{proj.get("title")}</div>
                                <div class="section-sub">📅 {proj.get("duration") or ""}</div>
                                <div class="section-desc">{proj.get("description") or ""}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No projects entries parsed.")
                    
            # Languages and Certifications
            col_lang, col_cert = st.columns(2)
            with col_lang:
                with st.expander("🗣️ Languages", expanded=True):
                    langs = parsed.get("languages", [])
                    if langs:
                        badges = "".join(f'<span class="lang-badge">{l}</span>' for l in langs)
                        st.markdown(badges, unsafe_allow_html=True)
                    else:
                        st.info("No languages parsed.")
            with col_cert:
                with st.expander("📜 Certifications", expanded=True):
                    certs = parsed.get("certifications", [])
                    if certs:
                        for cert in certs:
                            st.write(f"- {cert}")
                    else:
                        st.info("No certifications parsed.")

        # =========================================================================
        # TAB 2: SCREEN & MATCHING
        # =========================================================================
        with tab_match:
            analysis = st.session_state.analysis
            
            c_score, c_rec = st.columns([1, 2])
            
            with c_score:
                st.markdown('<div class="match-gauge-panel">', unsafe_allow_html=True)
                score = analysis.get("match_percentage", 0)
                st.markdown(f'<div class="match-score-text">{score}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="match-label">Match Percentage Score</div>', unsafe_allow_html=True)
                
                # Recommendation Badge Styling
                rec = analysis.get("hiring_recommendation", "Unsuitable")
                if rec == "Strong Fit":
                    st.markdown('<span class="recommendation-badge rec-strong">Strong Fit</span>', unsafe_allow_html=True)
                elif rec == "Moderate Fit":
                    st.markdown('<span class="recommendation-badge rec-moderate">Moderate Fit</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="recommendation-badge rec-unsuitable">Unsuitable</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c_rec:
                st.markdown('<div class="premium-card" style="height:100%;">', unsafe_allow_html=True)
                st.subheader("💡 Match Explanation")
                st.write(analysis.get("match_explanation", "No explanation available."))
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.markdown("---")
            
            # Gaps analysis
            c_strengths, c_weaknesses = st.columns(2)
            with c_strengths:
                st.markdown('<div class="premium-card" style="min-height:220px;">', unsafe_allow_html=True)
                st.subheader("✅ Candidate Strengths")
                strengths = analysis.get("strengths", [])
                if strengths:
                    for s in strengths:
                        st.write(f"- {s}")
                else:
                    st.write("*No strengths parsed or matching.*")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c_weaknesses:
                st.markdown('<div class="premium-card" style="min-height:220px;">', unsafe_allow_html=True)
                st.subheader("❌ Candidate Weaknesses / Gaps")
                weaknesses = analysis.get("weaknesses", [])
                if weaknesses:
                    for w in weaknesses:
                        st.write(f"- {w}")
                else:
                    st.write("*No notable gaps found.*")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Missing Skills Row
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("⚠️ Missing Skills Detection")
            st.write("Skills found in the Job Description that were not explicitly detected in the candidate's resume:")
            missing = analysis.get("missing_skills", [])
            if missing:
                badges = "".join(f'<span class="missing-skill-badge">{s}</span>' for s in missing)
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.success("No missing skills! The candidate matches the requested skillset catalog.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Interview Questions
            st.subheader("❓ Custom Interview Questions Generator")
            questions = st.session_state.questions
            
            q_tech, q_behav, q_hr = st.tabs(["💻 Technical Questions", "🤝 Behavioral (STAR)", "🏢 HR & Cultural Fit"])
            
            with q_tech:
                tech_q = questions.get("technical", [])
                if tech_q:
                    for i, q in enumerate(tech_q):
                        st.markdown(f"**Q{i+1}: {q.get('question')}**")
                        st.info(f"💡 *Ideal Answer Outline*: {q.get('ideal_answer_outline')}")
                        st.write("")
                else:
                    st.write("No technical questions generated.")
                    
            with q_behav:
                behav_q = questions.get("behavioral", [])
                if behav_q:
                    for i, q in enumerate(behav_q):
                        st.markdown(f"**Q{i+1}: {q.get('question')}**")
                        st.info(f"💡 *Ideal Answer Outline*: {q.get('ideal_answer_outline')}")
                        st.write("")
                else:
                    st.write("No behavioral questions generated.")
                    
            with q_hr:
                hr_q = questions.get("hr", [])
                if hr_q:
                    for i, q in enumerate(hr_q):
                        st.markdown(f"**Q{i+1}: {q.get('question')}**")
                        st.info(f"💡 *Ideal Answer Outline*: {q.get('ideal_answer_outline')}")
                        st.write("")
                else:
                    st.write("No HR/cultural questions generated.")

        # =========================================================================
        # TAB 3: AI RECRUITER CHATBOT (RAG)
        # =========================================================================
        with tab_chat:
            st.subheader("💬 Interactive Recruiter screening chatbot (RAG-Powered)")
            st.write(
                "Ask details regarding the candidate's achievements, project experience, "
                "missing skills, or alignment. The chatbot is strictly grounded in the "
                "uploaded documents."
            )
            
            # Render chat messages
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # If this message was generated by assistant, we display the sources if present
                    if message["role"] == "assistant" and "sources" in message and message["sources"]:
                        with st.expander("🔍 Grounding Sources (Document Chunks)"):
                            for i, src in enumerate(message["sources"]):
                                src_lbl = "Resume" if src["source"] == "resume" else "Job Description"
                                st.markdown(f"**Source {i+1} ({src_lbl} - Chunk #{src.get('index', 0)}):**")
                                st.markdown(f"*{src['text']}*")
                                st.markdown("---")
            
            # Input query
            query = st.chat_input("Ask a question about the candidate or job alignment...")
            
            if query:
                # Add user query to chat history
                st.session_state.chat_history.append({"role": "user", "content": query})
                
                # Render query
                with st.chat_message("user"):
                    st.markdown(query)
                    
                # RAG response execution
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving relevant documents and answering..."):
                        answer, sources = generate_rag_response(query, st.session_state.chat_history[:-1])
                        st.markdown(answer)
                        
                        if sources:
                            with st.expander("🔍 Grounding Sources (Document Chunks)"):
                                for i, src in enumerate(sources):
                                    src_lbl = "Resume" if src["source"] == "resume" else "Job Description"
                                    st.markdown(f"**Source {i+1} ({src_lbl} - Chunk #{src.get('index', 0)}):**")
                                    st.markdown(f"*{src['text']}*")
                                    st.markdown("---")
                                    
                # Add assistant reply to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
                # Trigger rerun to scroll properly
                st.rerun()

if __name__ == "__main__":
    main()