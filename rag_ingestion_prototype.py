









import os
import json
import tempfile
import shutil 
import base64
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder
from PIL import Image
import numpy as np
import easyocr

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- Pinecone Imports ---
from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# ------------------------------------------------------------
# DATABASE CONFIG & PINECONE INIT
# ------------------------------------------------------------
BASE_DB_DIR = "vector_db"
DB_TYPES = {
    "historical": {
        "meta": os.path.join(BASE_DB_DIR, "historical", "metadata.json")
    },
    "current": {
        "meta": os.path.join(BASE_DB_DIR, "current", "metadata.json")
    }
}

PINECONE_INDEX_NAME = "resume-db"

# Initialize Pinecone Client
# Initialize Pinecone Client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Ensure Index Exists
existing_indexes = pc.list_indexes().names()

if PINECONE_INDEX_NAME not in existing_indexes:
    try:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",          # ✅ VALID CLOUD
                region="us-east-1"    # ✅ VALID REGION
            )
        )
    except Exception as e:
        st.error(f"Error creating Pinecone index: {e}")


# ------------------------------------------------------------
# INITIALIZE LOCAL METADATA STORAGE
# ------------------------------------------------------------
def init_storage():
    for db_key, cfg in DB_TYPES.items():
        folder = os.path.dirname(cfg["meta"])
        Path(folder).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(cfg["meta"]):
            with open(cfg["meta"], "w") as f:
                json.dump({}, f)

# ------------------------------------------------------------
# EMBEDDING LOADER
# ------------------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)

# ------------------------------------------------------------
# METADATA OPERATIONS
# ------------------------------------------------------------
def load_metadata(db_type):
    try:
        with open(DB_TYPES[db_type]["meta"], "r") as f:
            return json.load(f)
    except:
        return {}

def save_metadata(meta, db_type):
    with open(DB_TYPES[db_type]["meta"], "w") as f:
        json.dump(meta, f, indent=2)

# ------------------------------------------------------------
# PINECONE VECTORSTORE OPERATIONS
# ------------------------------------------------------------
def get_vectorstore(embeddings, namespace):
    index = pc.Index(PINECONE_INDEX_NAME)
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace
    )

# ------------------------------------------------------------
# ADD DOCUMENTS TO VECTORSTORE
# ------------------------------------------------------------
def add_to_vectordb(docs, embeddings, db_type, file_id, original_name, meta_entry):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    for s in splits:
        s.metadata.update({
            "resume_id": file_id,
            "original_name": original_name,
            "db_type": db_type
        })

    vs = get_vectorstore(embeddings, db_type)
    vs.add_documents(splits)

    meta = load_metadata(db_type)
    meta[file_id] = meta_entry
    save_metadata(meta, db_type)

# ------------------------------------------------------------
# PROCESS FILE UPLOAD
# ------------------------------------------------------------
def process_upload(uploaded_file, embeddings, target_db="historical"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = f"{timestamp}_{uploaded_file.name}"

    ext = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # ---------- PDF ----------
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

        # ---------- TEXT ----------
        elif ext in ["txt"]:
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()

        # ---------- IMAGE (JPG / PNG) ----------
        elif ext in ["jpg", "jpeg", "png"]:
            reader = load_ocr_reader()
            image = np.array(Image.open(tmp_path).convert("RGB"))
            ocr_text = "\n".join(reader.readtext(image, detail=0))

            docs = [
                Document(
                    page_content=ocr_text,
                    metadata={"source": uploaded_file.name, "file_type": ext}
                )
            ]

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        full_text = "\n".join([d.page_content for d in docs])

        meta_entry = {
            "original_name": uploaded_file.name,
            "upload_date": timestamp,
            "file_size": uploaded_file.size,
            "preview": full_text[:300],
            "name": "-",
            "surname": "-",
            "phone": "-",
            "email": "-"
        }

        add_to_vectordb(docs, embeddings, target_db, file_id, uploaded_file.name, meta_entry)
        return file_id, docs, full_text

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ------------------------------------------------------------
# RETRIEVE RESUME TEXT
# ------------------------------------------------------------
def get_resume_text(file_id, db_type, embeddings):
    vs = get_vectorstore(embeddings, db_type)
    results = vs.similarity_search("", k=100, filter={"resume_id": file_id})
    if not results:
        return None
    return "\n\n".join([d.page_content for d in results])

# ------------------------------------------------------------
# DELETE RESUME
# ------------------------------------------------------------
def delete_from_db(file_id, db_type, embeddings):
    meta = load_metadata(db_type)
    if file_id in meta:
        del meta[file_id]
        save_metadata(meta, db_type)

    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        index.delete(filter={"resume_id": file_id}, namespace=db_type)
    except Exception as e:
        st.error(f"Error deleting from Pinecone: {e}")

# ------------------------------------------------------------
# CLEAR DATABASE
# ------------------------------------------------------------
def clear_database(db_type):
    save_metadata({}, db_type)
    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=db_type)
    except Exception as e:
        st.error(f"Error clearing Pinecone namespace: {e}")

# ------------------------------------------------------------
# LOAD LLM
# ------------------------------------------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

# ------------------------------------------------------------
# SHORTLIST EVALUATION LOGIC
# ------------------------------------------------------------
SHORTLIST_PROMPT = """
You are an expert HR AI. Evaluate the resume against the job description. 

Rules:
1. Score from 0-100.
2. If score >= 60 → Shortlisted else Rejected.
3. Extract: name, surname, email, phone.
4. also mention the tech role with respect to jd and store in the place of role of postion.
5. Return ONLY JSON.

Format:
{{
  "name": "",
  "surname": "",
  "email": "",
  "phone": "",
  "score": "0",
  "decision": "",
  "reason": "",
  "role of position":"",
}}

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd}
"""

def evaluate_candidate(llm, resume_text, jd):
    prompt = ChatPromptTemplate.from_template(SHORTLIST_PROMPT)
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({"resume_text": resume_text, "jd": jd})
        if not response or "{" not in response:
            return None
        
        start = response.find("{")
        end = response.rfind("}") + 1
        cleaned = response[start:end]
        return json.loads(cleaned)
    except Exception as e:
        st.error(f"LLM error: {e}")
        return None

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------
def load_custom_css():
    st.markdown("""
    <style>

    /* ------------------ GLOBAL BACKGROUND ------------------ */
    .stApp {
        background: linear-gradient(135deg, #F5F7FB 0%, #EEF2F7 100%);
        color: #1F2937;
    }

    /* ------------------ PAGE TITLES ------------------ */
    .page-title {
        font-size: 34px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
    }

    .page-subtitle {
        font-size: 15px;
        color: #64748B;
        margin-bottom: 24px;
    }

    /* ------------------ SIDEBAR ------------------ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E9EEF6, #F4F7FB);
        border-right: 1px solid #D6DCE6;
    }

    [data-testid="stSidebar"] .stRadio label {
        font-size: 15px;
        font-weight: 600;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 6px;
        transition: all 0.2s ease-in-out;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #FFE9D6;
        color: #C2410C;
    }

    [data-testid="stSidebar"] .stRadio label[data-selected="true"] {
        background-color: #FED7AA;
        color: #9A3412;
        font-weight: 700;
    }

    /* ------------------ CARDS ------------------ */
    .card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0px 10px 24px rgba(0, 0, 0, 0.05);
    }

    /* ------------------ KPI CARDS ------------------ */
    .kpi-card {
        background: linear-gradient(180deg, #FFFFFF, #FAFBFF);
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0px 12px 28px rgba(15, 23, 42, 0.08);
        text-align: center;
        transition: transform 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
    }

    .kpi-title {
        font-size: 14px;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 34px;
        font-weight: 800;
        color: #0F172A;
    }

    /* ------------------ BUTTONS ------------------ */
    button[kind="primary"] {
        background: linear-gradient(135deg, #F97316, #EA580C);
        color: white;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 700;
        border: none;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #EA580C, #C2410C);
    }

    /* ------------------ SECTION HEADERS ------------------ */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 30px;
        margin-bottom: 14px;
        border-left: 5px solid #F97316;
        padding-left: 12px;
    }

    /* ------------------ BADGES ------------------ */
    .badge-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
    }

    .badge-mid {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
    }

    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
    }
/* -------- GLASS SIDEBAR -------- */
[data-testid="stSidebar"] {
    background: rgba(236, 241, 248, 0.75);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-right: 1px solid rgba(203, 213, 225, 0.6);
}
/* -------- SIDEBAR ICON HOVER GLOW -------- */
[data-testid="stSidebar"] .stRadio label:hover {
    box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.15),
                0 4px 14px rgba(249, 115, 22, 0.25);
}
/* -------- ANIMATED PROGRESS BAR -------- */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(
        90deg,
        #F97316,
        #FDBA74,
        #F97316
    );
    background-size: 200% 100%;
    animation: progressFlow 1.5s linear infinite;
}

@keyframes progressFlow {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

    </style>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
def sidebar_logo():
    logo_path = "nx_logo.png"

    if os.path.exists(logo_path):
        logo_encoded = base64.b64encode(
            open(logo_path, "rb").read()
        ).decode()

        st.sidebar.markdown(f"""
        <div style="text-align:center; margin-bottom:25px;">
            <img src="data:image/png;base64,{logo_encoded}"
                 style="width:120px; margin-bottom:10px;" />
            <div style="font-size:18px; font-weight:800; color:#1E3A8A;">
                NeuroInkX
            </div>
            <div style="font-size:13px; color:#64748B;">
                HR AI Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown(
        "<div class='sidebar-title'>Navigation</div>",
        unsafe_allow_html=True
    )

    return st.sidebar.radio(
        "",
        [
            "🏠 Dashboard",
            "📤 Upload Resumes",
            "📊 Shortlist Candidates",
            "📁 Manage Databases",
            "📩 Quick Hire"
            "⚙️ Settings"
        ]
    )

# PAGE 1: DASHBOARD
# ------------------------------------------------------------
def render_dashboard_home():
    load_custom_css()

    # -------------------- HEADER --------------------
    st.markdown("""
    <div class="page-title">HR Analytics Dashboard</div>
    <div class="page-subtitle">
        Real-time overview of hiring performance & AI screening results
    </div>
    """, unsafe_allow_html=True)

    # -------------------- DATA --------------------
    hist_meta = load_metadata("historical")
    curr_meta = load_metadata("current")

    today = datetime.now().strftime("%Y%m%d")
    today_count = sum(
        1 for v in hist_meta.values()
        if v.get("upload_date", "").startswith(today)
    )

    total_resumes = len(hist_meta)
    shortlisted = len(curr_meta)
    rejected = max(0, total_resumes - shortlisted)

    # -------------------- KPI CARDS --------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📄 Total Resumes</div>
            <div class="kpi-value">{total_resumes}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">✅ Shortlisted</div>
            <div class="kpi-value">{shortlisted}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📅 Uploaded Today</div>
            <div class="kpi-value">{today_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------- CHARTS --------------------
    col_left, col_right = st.columns([2, 3])

    # Pie Chart
    with col_left:
        st.markdown("### 📊 Shortlisting Distribution")
        if shortlisted + rejected > 0:
            fig = px.pie(
                names=["Shortlisted", "Rejected"],
                values=[shortlisted, rejected],
                hole=0.5
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shortlisting data available yet.")

    # Upload Trend
    with col_right:
        st.markdown("### 📈 Resume Upload Trend")
        upload_dates = [v["upload_date"][:8] for v in hist_meta.values() if "upload_date" in v]
        if upload_dates:
            df = pd.DataFrame(upload_dates, columns=["date"])
            upload_trend = df["date"].value_counts().sort_index()
            fig2 = px.line(
                x=upload_trend.index,
                y=upload_trend.values,
                markers=True
            )
            fig2.update_layout(
                xaxis_title="Date",
                yaxis_title="Resumes Uploaded"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No upload trend data available.")

def render_upload_page(embeddings):
    load_custom_css()
# page 2 uplaod resumes 
    # -------------------- HEADER --------------------
    st.markdown("""
    <div class="page-title">📥 Resume Intake Center</div>
    <div class="page-subtitle">
        Centralized resume upload with AI parsing & OCR support
    </div>
    """, unsafe_allow_html=True)

    # -------------------- INFO BAR --------------------
    st.markdown("""
    <div class="info-box">
        <b>Supported Formats:</b> PDF, TXT, JPG, JPEG, PNG <br>
        <b>Image resumes:</b> OCR is applied automatically <br>
        <b>Tip:</b> Clear, high-resolution images give better extraction accuracy
    </div>
    """, unsafe_allow_html=True)

    # -------------------- UPLOAD CARD --------------------
    st.markdown("""
    <div class="card">
        <h3>Upload Resumes</h3>
        <p>
        Upload resumes into the <b>Current Database</b>.  
        After shortlisting, candidates can be archived into the <b>Historical Database</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "📂 Drag & drop or browse files",
        type=["pdf", "txt", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="resume_uploader"
    )

    if uploaded_files:
        st.markdown(f"""
        <div class="upload-stats">
            📄 Files selected: <b>{len(uploaded_files)}</b>
        </div>
        """, unsafe_allow_html=True)

    # -------------------- PROCESS BUTTON --------------------
    process = st.button("🚀 Process Uploads", type="primary", key="process_upload_button")

    if process:
        if not uploaded_files:
            st.warning("Please upload at least one resume.")
            return

        existing_meta = load_metadata("current")
        existing_names = {m["original_name"].lower() for m in existing_meta.values()}

        duplicates, new_files = [], []
        for f in uploaded_files:
            (duplicates if f.name.lower() in existing_names else new_files).append(f)

        # -------------------- DUPLICATE INFO --------------------
        if duplicates:
            st.warning(
                "⚠️ Duplicate resumes skipped:\n\n" +
                "\n".join([f"• {x.name}" for x in duplicates])
            )

        if not new_files:
            st.error("All uploaded resumes already exist in the CURRENT database.")
            return

        st.markdown(f"""
        <div class="upload-stats">
            ✅ New files to process: <b>{len(new_files)}</b>
        </div>
        """, unsafe_allow_html=True)

        # -------------------- PROGRESS --------------------
        progress = st.progress(0)
        uploaded_info = []

        for idx, f in enumerate(new_files):
            try:
                file_id, docs, text = process_upload(f, embeddings, "current")
                uploaded_info.append({
                    "File Name": f.name,
                    "Preview": text[:200] + "..."
                })
            except Exception as e:
                st.error(f"❌ Error processing {f.name}: {e}")

            progress.progress((idx + 1) / len(new_files))

        st.success("🎉 Upload complete! Resumes added to CURRENT database.")

        # -------------------- SUMMARY TABLE --------------------
        if uploaded_info:
            st.markdown("<h3 class='section-header'>📋 Upload Summary</h3>", unsafe_allow_html=True)

            df = pd.DataFrame(uploaded_info)
            builder = GridOptionsBuilder.from_dataframe(df)
            builder.configure_default_column(editable=False)
            builder.configure_pagination(enabled=True)

            AgGrid(
                df,
                gridOptions=builder.build(),
                height=320,
                theme="balham",
                key="upload_summary_grid"
            )

def render_shortlist_page(llm, embeddings):
    load_custom_css()
# page 3 ai shoetlisting 
    # ------------------------------------------------------------
# PAGE 3: AI SHORTLISTING (FIXED & POLISHED)
# ------------------------------------------------------------
def render_shortlist_page(llm, embeddings):
    load_custom_css()

    # ===================== HEADER =====================
    st.markdown("""
    <div class="page-title">🤖 AI Resume Shortlisting</div>
    <div class="page-subtitle">
        AI-powered evaluation of resumes against your Job Description
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <b>How it works</b><br>
        • AI reads resumes (PDF / TXT / OCR Images)<br>
        • Matches skills with Job Description<br>
        • Generates score, decision & reasoning
    </div>
    """, unsafe_allow_html=True)

    # ===================== JOB DESCRIPTION =====================
    st.markdown("<h3 class='section-header'>📄 Job Description</h3>", unsafe_allow_html=True)

    template_option = st.selectbox(
        "Choose a JD Template",
        [
            "Write JD manually",
            "Software Engineer",
            "Data Analyst",
            "Backend Developer",
            "Flutter Developer",
            "Product Manager",
            "DevOps Engineer"
        ]
    )

    jd_presets = {
        "Software Engineer": "We need a Software Engineer skilled in DSA, Python/Java, OOP, Git, SQL, APIs, SDLC.",
        "Data Analyst": "Looking for a Data Analyst skilled in Excel, SQL, Python, dashboards, insights.",
        "Backend Developer": "Backend Engineer with Node.js, REST APIs, MongoDB/MySQL, authentication.",
        "Flutter Developer": "Flutter Developer with Dart, state management, REST APIs, UI/UX.",
        "Product Manager": "PM with PRDs, roadmaps, KPIs, user research, stakeholder communication.",
        "DevOps Engineer": "DevOps Engineer skilled in AWS, Docker, CI/CD, Kubernetes, monitoring."
    }

    jd = st.text_area(
        "Enter Job Description",
        jd_presets.get(template_option, ""),
        height=160
    )

    # ===================== CANDIDATE SOURCE =====================
    st.markdown("<h3 class='section-header'>👥 Candidate Pool</h3>", unsafe_allow_html=True)

    db_key = "current"
    metadata = load_metadata(db_key)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📂 Candidates available: **{len(metadata)}**")
    with col2:
        run_eval = st.button("🚀 Run AI Evaluation", type="primary")

    # ===================== RUN AI EVALUATION =====================
    if run_eval:
        if not jd or not metadata:
            st.error("Please provide a Job Description and ensure resumes exist.")
            return

        results = []
        total = len(metadata)

        with st.status("🤖 AI is evaluating resumes...", expanded=True) as status:
            progress = st.progress(0)

            for idx, (file_id, info) in enumerate(metadata.items()):
                resume_text = get_resume_text(file_id, db_key, embeddings)
                if not resume_text:
                    continue

                eval_result = evaluate_candidate(llm, resume_text, jd)
                if not eval_result:
                    continue

                # -------- SAFE SCORE NORMALIZATION --------
                try:
                    score = int(float(eval_result.get("score", 0)))
                except:
                    score = 0

                if score >= 75:
                    badge = "<span class='badge-pass'>Excellent</span>"
                    decision = "Shortlisted"
                elif score >= 60:
                    badge = "<span class='badge-mid'>Good</span>"
                    decision = "Shortlisted"
                else:
                    badge = "<span class='badge-low'>Poor</span>"
                    decision = "Rejected"

                info.update({
                    "name": eval_result.get("name", "-"),
                    "surname": eval_result.get("surname", "-"),
                    "phone": eval_result.get("phone", "-"),
                    "email": eval_result.get("email", "-")
                })
                save_metadata(metadata, db_key)

                results.append({
                    "Name": info["name"],
                    "Surname": info["surname"],
                    "Email": info["email"],
                    "Phone": info["phone"],
                    "Score": score,
                    "Decision": decision,
                    "Reason": eval_result.get("reason", ""),
                    "Role": eval_result.get("role of position", ""),
                    "Badge": badge,
                    "Resume ID": file_id,
                    "Text": resume_text
                })

                progress.progress((idx + 1) / total)

            status.update(
                label="✅ Evaluation completed successfully",
                state="complete"
            )

        if not results:
            st.error("No candidates evaluated.")
            return

        st.session_state["shortlist_results"] = results
        st.success("🎉 AI Evaluation Completed Successfully")

    # ===================== RESULTS VIEW =====================
    if "shortlist_results" in st.session_state:
        results = st.session_state["shortlist_results"]

        scores = [r["Score"] for r in results]
        shortlisted = len([s for s in scores if s >= 60])
        rejected = len(scores) - shortlisted

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>📄 Evaluated</div><div class='kpi-value'>{len(scores)}</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>✅ Shortlisted</div><div class='kpi-value'>{shortlisted}</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>❌ Rejected</div><div class='kpi-value'>{rejected}</div></div>", unsafe_allow_html=True)

        st.markdown("<h3 class='section-header'>📊 Candidate Results</h3>", unsafe_allow_html=True)

        for r in results:
            st.markdown(f"""
            <div class="candidate-card">
                <div class="candidate-header">
                    <span>{r['Name']} {r['Surname']}</span>
                    {r['Badge']}
                </div>
                <div class="candidate-meta">
                    📧 {r['Email']} &nbsp; | &nbsp; 📞 {r['Phone']}<br>
                    🎯 Role Match: <b>{r['Role']}</b>
                </div>
                <div class="candidate-score">
                    Score: <b>{r['Score']}</b> / 100 — <b>{r['Decision']}</b>
                </div>
                <div class="candidate-reason">
                    <b>AI Reason:</b> {r['Reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📄 View Resume Text"):
                st.text_area("Resume", r["Text"], height=260)

        # ===================== EXPORT =====================
        st.markdown("<h3 class='section-header'>📥 Export Results</h3>", unsafe_allow_html=True)

        df = pd.DataFrame(results)
        export_cols = ["Name", "Surname", "Email", "Phone", "Score", "Decision", "Reason", "Role"]
        df_export = df[[c for c in export_cols if c in df.columns]]

        st.download_button(
            "⬇️ Download CSV",
            df_export.to_csv(index=False).encode("utf-8"),
            "shortlist_results.csv",
            "text/csv"
        )

# ------------------------------------------------------------
# PAGE 4: MANAGE DATABASES (FIXED + PREMIUM UI)
# ------------------------------------------------------------
def render_manage_databases_page(embeddings):
    load_custom_css()

    # ===================== HEADER =====================
    st.markdown("""
    <div class="page-title">📁 Database Management</div>
    <div class="page-subtitle">
        Manage, archive, and clean resume databases safely
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📚 Historical Database", "⭐ Current Database"])

    # =================================================
    # INTERNAL RENDER FUNCTION
    # =================================================
    def render_db(db_key):
        meta = load_metadata(db_key)

        # ---------------- EMPTY STATE ----------------
        if not meta:
            st.markdown("""
            <div class="card">
                <b>No resumes found</b><br>
                Upload resumes to start managing this database.
            </div>
            """, unsafe_allow_html=True)
            return

        # ---------------- STATS ----------------
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📄 Total Resumes</div>
            <div class="kpi-value">{len(meta)}</div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- TABLE ----------------
        df = pd.DataFrame.from_dict(meta, orient="index")

        for col in ["original_name", "name", "surname", "email", "phone", "upload_date", "file_size"]:
            if col not in df.columns:
                df[col] = "-"

        view_df = df[
            ["original_name", "name", "surname", "email", "phone", "upload_date", "file_size"]
        ]

        view_df.columns = [
            "File Name",
            "First Name",
            "Last Name",
            "Email",
            "Phone",
            "Upload Date",
            "Size (bytes)"
        ]

        builder = GridOptionsBuilder.from_dataframe(view_df)
        builder.configure_default_column(editable=False, filter=True, sortable=True)
        builder.configure_pagination(enabled=True)
        builder.configure_selection("multiple")

        grid = AgGrid(
            view_df,
            gridOptions=builder.build(),
            height=400,
            theme="balham",
            key=f"{db_key}_grid"
        )

        selected = grid["selected_rows"]

        # ---------------- DELETE SELECTED ----------------
        st.markdown("<h3 class='section-header'>🗑️ Delete Selected Resumes</h3>", unsafe_allow_html=True)

        if st.button(f"Delete Selected from {db_key.title()}", type="secondary", key=f"delete_{db_key}"):
            if not selected:
                st.warning("Please select at least one resume.")
            else:
                for row in selected:
                    for fid, m in meta.items():
                        if m["original_name"] == row["File Name"]:
                            delete_from_db(fid, db_key, embeddings)
                st.success("Selected resumes deleted successfully.")
                st.rerun()

        # ---------------- ARCHIVE ----------------
        if db_key == "current":
            st.markdown("<h3 class='section-header'>📦 Archive Database</h3>", unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                Archiving moves all CURRENT resumes into the HISTORICAL database.<br>
                <b>This action cannot be undone.</b>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Archive Current → Historical", type="primary", key="archive_db"):
                progress_bar = st.progress(0)
                total_files = len(meta)

                for idx, (fid, info) in enumerate(meta.items()):
                    text = get_resume_text(fid, "current", embeddings)
                    if text:
                        doc = Document(
                            page_content=text,
                            metadata={"source": info["original_name"]}
                        )
                        add_to_vectordb(
                            [doc],
                            embeddings,
                            "historical",
                            fid,
                            info["original_name"],
                            info
                        )
                    progress_bar.progress((idx + 1) / total_files)

                clear_database("current")
                st.success("All resumes archived successfully.")
                st.rerun()

        # ---------------- DANGER ZONE ----------------
        st.markdown("<h3 class='section-header'>⚠️ Danger Zone</h3>", unsafe_allow_html=True)

        st.markdown("""
        <div class="danger-box">
            This will permanently delete <b>ALL</b> resumes from this database.
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            f"🔥 Clear Entire {db_key.title()} Database",
            type="secondary",
            key=f"clear_{db_key}"
        ):
            clear_database(db_key)
            st.success(f"{db_key.title()} database cleared permanently.")
            st.rerun()

    # ===================== TAB RENDER =====================
    with tabs[0]:
        render_db("historical")

    with tabs[1]:
        render_db("current")


# ------------------------------------------------------------
# PAGE 5: SETTINGS
# ------------------------------------------------------------
def render_settings_page():
    load_custom_css()

    # ===================== HEADER =====================
    st.markdown("""
    <div class="page-title">⚙️ Application Settings</div>
    <div class="page-subtitle">
        Configure AI behavior, scoring logic, and UI preferences
    </div>
    """, unsafe_allow_html=True)

    # ===================== AI MODEL =====================
    st.markdown("<h3 class='section-header'>🤖 AI Model Configuration</h3>", unsafe_allow_html=True)

    model = st.selectbox(
        "Select LLM Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.1-70b",
            "mixtral-8x7b",
            "gemma-7b"
        ],
        index=0
    )

    st.info(f"Selected model will apply after application restart: **{model}**")

    # ===================== SCORING =====================
    st.markdown("<h3 class='section-header'>📊 Shortlisting Thresholds</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        shortlist_threshold = st.slider(
            "Minimum score to shortlist",
            0, 100, 60
        )
    with col2:
        excellent_threshold = st.slider(
            "Excellent candidate score",
            0, 100, 75
        )

    st.markdown(f"""
    <div class="info-box">
        Candidates with score ≥ <b>{shortlist_threshold}</b> will be shortlisted.<br>
        Candidates with score ≥ <b>{excellent_threshold}</b> will be marked excellent.
    </div>
    """, unsafe_allow_html=True)

    # ===================== UI =====================
    st.markdown("<h3 class='section-header'>🎨 UI Preferences</h3>", unsafe_allow_html=True)

    theme = st.selectbox(
        "Select Theme",
        ["Dark (recommended)", "Light", "High Contrast"]
    )

    st.info(f"Theme will apply in next release: **{theme}**")

    # ===================== SAVE =====================
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("💾 Save Settings", type="primary")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Dual-DB Resume Shortlister", page_icon="🤖", layout="wide")
    load_custom_css()
    init_storage()
    llm = load_llm()
    embeddings = load_embeddings()
    menu = sidebar_logo()

    if menu == "🏠 Dashboard": render_dashboard_home()
    elif menu == "📤 Upload Resumes": render_upload_page(embeddings)
    elif menu == "📊 Shortlist Candidates": render_shortlist_page(llm, embeddings)
    elif menu == "📁 Manage Databases": render_manage_databases_page(embeddings)
    elif menu == "⚙️ Settings": render_settings_page()

if __name__ == "__main__":
    main()










