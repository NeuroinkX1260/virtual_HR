



# ###############################################################
# #  DUAL DATABASE RESUME SHORTLISTER — FULL FIXED VERSION
# ###############################################################
# # ------------------------------------------------------------
# # FULL PREMIUM UI UPGRADE FOR HR SYSTEM — PART 1
# # ------------------------------------------------------------

# import os
# import json
# import tempfile
# import shutil
# import base64
# import streamlit as st
# import pandas as pd
# from dotenv import load_dotenv
# from pathlib import Path
# from datetime import datetime

# # LangChain Imports
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.documents import Document

# # Load environment variables
# load_dotenv()

# # ------------------------------------------------------------
# # DATABASE CONFIG
# # ------------------------------------------------------------
# BASE_DB_DIR = "vector_db"
# DB_TYPES = {
#     "historical": {
#         "path": os.path.join(BASE_DB_DIR, "historical"),
#         "meta": os.path.join(BASE_DB_DIR, "historical", "metadata.json")
#     },
#     "current": {
#         "path": os.path.join(BASE_DB_DIR, "current"),
#         "meta": os.path.join(BASE_DB_DIR, "current", "metadata.json")
#     }
# }

# # ------------------------------------------------------------
# # INITIALIZE STORAGE DIRECTORIES
# # ------------------------------------------------------------
# def init_storage():
#     for db_key, cfg in DB_TYPES.items():
#         Path(cfg["path"]).mkdir(parents=True, exist_ok=True)
#         if not os.path.exists(cfg["meta"]):
#             with open(cfg["meta"], "w") as f:
#                 json.dump({}, f)

# # ------------------------------------------------------------
# # EMBEDDING LOADER (CACHED)
# # ------------------------------------------------------------
# @st.cache_resource
# def load_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )

# # ------------------------------------------------------------
# # METADATA OPERATIONS
# # ------------------------------------------------------------
# def load_metadata(db_type):
#     try:
#         with open(DB_TYPES[db_type]["meta"], "r") as f:
#             return json.load(f)
#     except:
#         return {}

# def save_metadata(meta, db_type):
#     with open(DB_TYPES[db_type]["meta"], "w") as f:
#         json.dump(meta, f, indent=2)

# # ------------------------------------------------------------
# # VECTORSTORE LOADING & SAVING
# # ------------------------------------------------------------
# def get_vectorstore(embeddings, db_type):
#     index_path = os.path.join(DB_TYPES[db_type]["path"], "faiss_index")
    
#     if os.path.exists(index_path):
#         try:
#             return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
#         except:
#             return None
#     return None

# def save_vectorstore(vs, db_type):
#     index_path = os.path.join(DB_TYPES[db_type]["path"], "faiss_index")
#     vs.save_local(index_path)

# # ------------------------------------------------------------
# # ADD DOCUMENTS TO VECTORSTORE
# # ------------------------------------------------------------
# def add_to_vectordb(docs, embeddings, db_type, file_id, original_name, meta_entry):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = splitter.split_documents(docs)

#     for s in splits:
#         s.metadata.update({
#             "resume_id": file_id,
#             "original_name": original_name,
#             "db_type": db_type
#         })

#     vs = get_vectorstore(embeddings, db_type)

#     if vs is None:
#         vs = FAISS.from_documents(splits, embeddings)
#     else:
#         vs.add_documents(splits)

#     save_vectorstore(vs, db_type)

#     meta = load_metadata(db_type)
#     meta[file_id] = meta_entry
#     save_metadata(meta, db_type)

# # ------------------------------------------------------------
# # PROCESS FILE UPLOAD
# # ------------------------------------------------------------
# def process_upload(uploaded_file, embeddings, target_db="historical"):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_id = f"{timestamp}_{uploaded_file.name}"

#     ext = uploaded_file.name.split(".")[-1].lower()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
#         tmp.write(uploaded_file.read())
#         tmp_path = tmp.name

#     try:
#         loader = PyPDFLoader(tmp_path) if ext == "pdf" else TextLoader(tmp_path, encoding="utf-8")
#         docs = loader.load()

#         full_text = "\n".join([d.page_content for d in docs])

#         meta_entry = {
#             "original_name": uploaded_file.name,
#             "upload_date": timestamp,
#             "file_size": uploaded_file.size,
#             "preview": full_text[:300],
#             "name": "-",
#             "surname": "-",
#             "phone": "-",
#             "email": "-"
#         }

#         add_to_vectordb(docs, embeddings, target_db, file_id, uploaded_file.name, meta_entry)

#         return file_id, docs, full_text

#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

# # ------------------------------------------------------------
# # COPY TO CURRENT DB
# # ------------------------------------------------------------
# def copy_to_current_db(file_id, docs, embeddings, meta_entry):
#     curr_meta = load_metadata("current")

#     if file_id in curr_meta:
#         return False  

#     add_to_vectordb(docs, embeddings, "current", file_id, meta_entry["original_name"], meta_entry)
#     return True

# # ------------------------------------------------------------
# # RETRIEVE RESUME TEXT FROM VECTORSTORE
# # ------------------------------------------------------------
# def get_resume_text(file_id, db_type, embeddings):
#     vs = get_vectorstore(embeddings, db_type)
#     if not vs:
#         return None

#     all_docs = vs.similarity_search("", k=5000)
#     docs = [d for d in all_docs if d.metadata.get("resume_id") == file_id]

#     if not docs:
#         return None

#     return "\n\n".join([d.page_content for d in docs])

# # ------------------------------------------------------------
# # DELETE RESUME FROM DB
# # ------------------------------------------------------------
# def delete_from_db(file_id, db_type, embeddings):
#     meta = load_metadata(db_type)
#     if file_id in meta:
#         del meta[file_id]
#         save_metadata(meta, db_type)

#     vs = get_vectorstore(embeddings, db_type)
#     if not vs:
#         return

#     all_docs = vs.similarity_search("", k=8000)
#     filtered = [d for d in all_docs if d.metadata.get("resume_id") != file_id]

#     if not filtered:
#         shutil.rmtree(os.path.join(DB_TYPES[db_type]["path"], "faiss_index"), ignore_errors=True)
#     else:
#         new_vs = FAISS.from_documents(filtered, embeddings)
#         save_vectorstore(new_vs, db_type)

# # ------------------------------------------------------------
# # CLEAR ENTIRE DATABASE
# # ------------------------------------------------------------
# def clear_database(db_type):
#     shutil.rmtree(DB_TYPES[db_type]["path"], ignore_errors=True)
#     init_storage()

# # ------------------------------------------------------------
# # LOAD LLM
# # ------------------------------------------------------------
# @st.cache_resource
# def load_llm():
#     return ChatGroq(
#         model_name="llama-3.1-8b-instant",
#         api_key=os.getenv("GROQ_API_KEY"),
#         temperature=0
#     )

# # ------------------------------------------------------------
# # SHORTLIST EVALUATION LOGIC
# # ------------------------------------------------------------
# SHORTLIST_PROMPT = """
# You are an expert HR AI. Evaluate the resume against the job description.

# Rules:
# 1. Score from 0-100.
# 2. If score >= 60 → Shortlisted else Rejected.
# 3. Extract: name, surname, email, phone.
# 4. Return ONLY JSON.

# Format:
# {{
#   "name": "",
#   "surname": "",
#   "email": "",
#   "phone": "",
#   "score": 0,
#   "decision": "",
#   "reason": ""
# }}

# RESUME:
# {resume_text}

# JOB DESCRIPTION:
# {jd}
# """




# def evaluate_candidate(llm, resume_text, jd):
#     prompt = ChatPromptTemplate.from_template(SHORTLIST_PROMPT)
#     chain = prompt | llm | StrOutputParser()

#     try:
#         response = chain.invoke({"resume_text": resume_text, "jd": jd})
#         if not response or "{" not in response:
#             return None
        
#         start = response.find("{")
#         end = response.rfind("}") + 1
#         cleaned = response[start:end]

#         return json.loads(cleaned)

#     except Exception as e:
#         st.error(f"LLM error: {e}")
#         return None

# # ------------------------------------------------------------
# # UI THEME + SIDEBAR + DASHBOARD HOME
# # ------------------------------------------------------------

# import plotly.express as px
# from st_aggrid import AgGrid, GridOptionsBuilder

# # ------------------------------------------------------------
# # CUSTOM CSS STYLING
# # ------------------------------------------------------------
# def load_custom_css():
#     st.markdown("""
#     <style>

#     /* Sidebar Container */
#     [data-testid="stSidebar"] {
#         background-color: #E8F1FA !important;  
#         padding-top: 30px !important;
#         border-right: 1px solid #CFD9E6 !important; 
#     }

#     /* Logo inside sidebar */
#     .sidebar-logo img {
#         width: 120px !important;
#         border-radius: 50% !important;
#         display: block !important;
#         margin-left: auto !important;
#         margin-right: auto !important;
#         margin-top: 10px !important;
#         margin-bottom: 25px !important;
#     }

#     /* Navigation Title */
#     .sidebar-title {
#         color: #1E3A8A !important;  
#         font-size: 20px !important;
#         font-weight: 700 !important;
#         padding-left: 20px !important;
#         margin-bottom: 20px !important;
#     }

#     /* Navigation Radio Buttons Text */
#     .sidebar .stRadio label {
#         color: #1F2937 !important;  
#         font-size: 16px !important;
#         font-weight: 600 !important;
#         padding: 5px 10px !important;
#         border-radius: 6px !important;
#     }

#     /* Hover Effect */
#     [data-testid="stSidebar"] .stRadio label:hover {
#         background-color: #DCEBFA !important;
#         color: #1E40AF !important;
#     }

#     /* Active Selected Option */
#     [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-selected="true"] {
#         background-color: #D0E4FF !important;
#         color: #1E3A8A !important;
#         font-weight: 700 !important;
#         border-radius: 6px !important;
#         padding: 6px 12px !important;
#     }

#     /* Sidebar Icons */
#     .sidebar .element-container svg {
#         color: #1E3A8A !important;
#         opacity: 1 !important;
#     }

#     </style>
#     """, unsafe_allow_html=True)


# # ------------------------------------------------------------
# # SIDEBAR NAVIGATION
# # ------------------------------------------------------------
# import base64

# def sidebar_logo():
#     logo_path = "nx.png"

#     logo_encoded = base64.b64encode(open(logo_path, "rb").read()).decode()

#     st.sidebar.markdown(
#         f"""
#         <div class="sidebar-logo">
#             <img src="data:image/png;base64,{logo_encoded}">
#         </div>
#         """,
#         unsafe_allow_html=True
#     )


#     st.sidebar.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)

#     return st.sidebar.radio(
#         "",
#         [
#             "🏠 Dashboard",
#             "📤 Upload Resumes",
#             "📊 Shortlist Candidates",
#             "📁 Manage Databases",
#             "⚙️ Settings"
#         ]
#     )

# # ------------------------------------------------------------
# # DASHBOARD HOME PAGE (ANALYTICS)
# # ------------------------------------------------------------
# def render_dashboard_home():
#     load_custom_css()

#     st.markdown("<h1 class='main-title'>HR Analytics Dashboard</h1>", unsafe_allow_html=True)
#     st.markdown("<div class='card'>A complete overview of your hiring pipeline and screening activity.</div>", unsafe_allow_html=True)

#     # Load metadata
#     hist_meta = load_metadata("historical")
#     curr_meta = load_metadata("current")

#     # Summary metrics
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Resumes", len(hist_meta))
#     with col2:
#         st.metric("Shortlisted (Current DB)", len(curr_meta))
#     with col3:
#         today = datetime.now().strftime("%Y%m%d")
#         today_count = sum(1 for v in hist_meta.values() if v["upload_date"].startswith(today))
#         st.metric("Uploaded Today", today_count)

#     # Pie chart for shortlisted vs rejected
#     shortlisted = len(curr_meta)
#     rejected = max(0, len(hist_meta) - shortlisted)

#     fig = px.pie(
#         names=["Shortlisted", "Rejected"],
#         values=[shortlisted, rejected],
#         title="Shortlisting Distribution"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # Upload trends chart
#     upload_dates = [
#         v["upload_date"][:8]  # extract YYYYMMDD
#         for v in hist_meta.values()
#     ]

#     if upload_dates:
#         df = pd.DataFrame(upload_dates, columns=["date"])
#         upload_trend = df["date"].value_counts().sort_index()

#         fig2 = px.line(
#             x=upload_trend.index,
#             y=upload_trend.values,
#             title="Resume Upload Trend"
#         )
#         fig2.update_layout(xaxis_title="Date", yaxis_title="Uploads")
#         st.plotly_chart(fig2, use_container_width=True)
# # ------------------------------------------------------------
# # UPLOAD RESUMES — MODERN UI + DUPLICATE CHECKER
# # ------------------------------------------------------------
# # ------------------------------------------------------------
# # ------------------------------------------------------------
# # UPLOAD RESUMES — MODERN UI + DUPLICATE CHECKER
# # ------------------------------------------------------------
# def render_upload_page(embeddings):
#     load_custom_css()

#     st.markdown("<h1 class='main-title'>📤 Upload Resumes</h1>", unsafe_allow_html=True)
#     st.markdown(
#         "<div class='card'>Upload resumes into the CURRENT database. "
#         "After shortlisting, HR can archive them into the Historical Database.</div>",
#         unsafe_allow_html=True
#     )

#     # FIXED — unique key added
#     uploaded_files = st.file_uploader(
#         "Drag & drop PDF/TXT files",
#         type=["pdf", "txt"],
#         accept_multiple_files=True,
#         key="resume_uploader"
#     )

#     if st.button("Process Uploads", type="primary", key="process_upload_button"):
#         if not uploaded_files:
#             st.warning("Please upload at least one resume.")
#             return

#         # CHECK DUPLICATES ONLY IN CURRENT DB
#         existing_meta = load_metadata("current")
#         existing_names = {m["original_name"].lower() for m in existing_meta.values()}

#         duplicates = []
#         new_files = []

#         for f in uploaded_files:
#             if f.name.lower() in existing_names:
#                 duplicates.append(f.name)
#             else:
#                 new_files.append(f)

#         if duplicates:
#             st.warning("⚠️ Duplicate resumes skipped:\n" + "\n".join([f"- {x}" for x in duplicates]))

#         if not new_files:
#             st.error("All uploaded resumes were duplicates in CURRENT DB.")
#             return

#         progress = st.progress(0)
#         uploaded_info = []

#         for idx, f in enumerate(new_files):
#             try:
#                 file_id, docs, text = process_upload(f, embeddings, "current")
#                 uploaded_info.append({
#                     "File Name": f.name,
#                     "Preview": text[:200] + "..."
#                 })
#             except Exception as e:
#                 st.error(f"Error processing {f.name}: {e}")

#             progress.progress((idx + 1) / len(new_files))

#         st.success("🎉 Upload complete! Added to CURRENT database.")

#         # SHOW SUMMARY
#         if uploaded_info:
#             st.markdown("<h3 class='section-header'>Uploaded Resumes Summary</h3>", unsafe_allow_html=True)

#             df = pd.DataFrame(uploaded_info)

#             builder = GridOptionsBuilder.from_dataframe(df)
#             builder.configure_default_column(editable=False)
#             builder.configure_pagination(enabled=True)
#             builder.configure_selection("single")

#             AgGrid(
#                 df,
#                 gridOptions=builder.build(),
#                 height=300,
#                 theme="balham",
#                 key="upload_summary_grid"
#             )

# # ------------------------------------------------------------
# # AI SHORTLISTING PAGE (PROFESSIONAL UI + AG-GRID + VIEWER)
# # ------------------------------------------------------------
# def render_shortlist_page(llm, embeddings):
#     load_custom_css()

#     st.markdown("<h1 class='main-title'>📊 AI Resume Shortlisting</h1>", unsafe_allow_html=True)
#     st.markdown("""
#         <div class='card'>
#             The AI evaluates each resume against the provided Job Description and generates a score,
#             decision, and reasoning. HR can export the report and later archive the Current Database.
#         </div>
#     """, unsafe_allow_html=True)

#     # --------------------------------------------------------
#     # JD Templates
#     # --------------------------------------------------------
#     st.markdown("<h3 class='section-header'>Job Description</h3>", unsafe_allow_html=True)

#     template_option = st.selectbox(
#         "Select a JD Template",
#         [
#             "Write JD manually",
#             "Software Engineer",
#             "Data Analyst",
#             "Backend Developer",
#             "Flutter Developer",
#             "Product Manager",
#             "DevOps Engineer"
#         ]
#     )

#     jd_presets = {
#         "Software Engineer": "We need a Software Engineer skilled in DSA, Python/Java, OOP, Git, SQL, API development, and SDLC.",
#         "Data Analyst": "We seek a Data Analyst skilled in Excel, SQL, Python, dashboards, visualizations, and reporting.",
#         "Backend Developer": "We need a Backend Engineer skilled in Node.js, Express, REST APIs, MongoDB/MySQL, and authentication.",
#         "Flutter Developer": "We need a Flutter Developer with Dart skills, REST API integration, state management, and UI/UX.",
#         "Product Manager": "We need a PM experienced in PRDs, roadmaps, KPIs, user interviews, and cross-team communication.",
#         "DevOps Engineer": "We seek a DevOps Engineer skilled in AWS, Docker, CI/CD pipelines, Kubernetes, and monitoring."
#     }

#     if template_option == "Write JD manually":
#         jd = st.text_area("Enter JD", height=150)
#     else:
#         jd = st.text_area("Enter JD", jd_presets[template_option], height=150)

#     # --------------------------------------------------------
#     # FIXED: ALWAYS EVALUATE ONLY CURRENT DB
#     # --------------------------------------------------------
#     st.markdown("<h3 class='section-header'>Candidate Source</h3>", unsafe_allow_html=True)
#     st.info("Shortlisting will be performed on **Current Database** only.")

#     db_key = "current"
#     metadata = load_metadata("current")

#     st.info(f"Total candidates in Current Database: **{len(metadata)}**")

#     # --------------------------------------------------------
#     # START EVALUATION
#     # --------------------------------------------------------
#     if st.button("🚀 Run AI Evaluation", type="primary"):

#         if not jd:
#             st.warning("Please enter a Job Description.")
#             return

#         if not metadata:
#             st.error("No candidates available in Current Database.")
#             return

#         progress = st.progress(0)
#         vs = get_vectorstore(embeddings, db_key)

#         results = []
#         total = len(metadata)

#         # ---------------------- MAIN LOOP ----------------------
#         for idx, (file_id, info) in enumerate(metadata.items()):

#             resume_text = get_resume_text(file_id, db_key, embeddings)
#             if not resume_text:
#                 continue

#             eval_result = evaluate_candidate(llm, resume_text, jd)
#             if not eval_result:
#                 continue

#             score = eval_result.get("score", 0)

#             # Score badge
#             if score >= 75:
#                 badge = "<span class='badge-pass'>Excellent</span>"
#             elif score >= 60:
#                 badge = "<span class='badge-mid'>Good</span>"
#             else:
#                 badge = "<span class='badge-low'>Poor</span>"

#             # Update metadata (only inside CURRENT DB)
#             info["name"] = eval_result.get("name", "-")
#             info["surname"] = eval_result.get("surname", "-")
#             info["phone"] = eval_result.get("phone", "-")
#             info["email"] = eval_result.get("email", "-")
#             save_metadata(metadata, db_key)

#             # --------------------------------------
#             # REMOVED auto-promotion (no more needed)
#             # --------------------------------------
#             promoted = False

#             # Add to results table
#             results.append({
#                 "Name": info["name"],
#                 "Surname": info["surname"],
#                 "Email": info["email"],
#                 "Phone": info["phone"],
#                 "Score": score,
#                 "Badge": badge,
#                 "Decision": eval_result.get("decision", ""),
#                 "Reason": eval_result.get("reason", ""),
#                 "File Name": info["original_name"],
#                 "Resume ID": file_id,
#                 "Promoted": "No",     # always No now
#                 "Text": resume_text
#             })

#             progress.progress((idx + 1) / total)

#         if not results:
#             st.error("No results generated.")
#             return

#         # --------------------------------------------------------
#         # DISPLAY TABLE
#         # --------------------------------------------------------
#         st.success("🎉 Evaluation Complete!")

#         df = pd.DataFrame(results)

#         st.markdown("<h3 class='section-header'>Evaluation Results</h3>", unsafe_allow_html=True)

#         builder = GridOptionsBuilder.from_dataframe(df)
#         builder.configure_pagination(enabled=True)
#         builder.configure_default_column(editable=False, filter=True, sortable=True)
#         builder.configure_column("Badge", cellRenderer="html")
#         builder.configure_selection("single")

#         grid = AgGrid(
#             df,
#             gridOptions=builder.build(),
#             height=420,
#             theme="material",
#             update_mode="MODEL_CHANGED"
#         )

#         selected = grid["selected_rows"]

#         # --------------------------------------------------------
#         # RESUME VIEWER
#         # --------------------------------------------------------
#         if selected:
#             st.markdown("<h3 class='section-header'>Resume Viewer</h3>", unsafe_allow_html=True)

#             row = selected[0]

#             st.write(f"### {row['Name']} {row['Surname']}")
#             st.write(f"📧 {row['Email']}")
#             st.write(f"📞 {row['Phone']}")
#             st.markdown(row["Badge"], unsafe_allow_html=True)
#             st.write("---")
#             st.text_area("Resume Text", row["Text"], height=350)

#         # --------------------------------------------------------
#         # EXPORT BUTTONS
#         # --------------------------------------------------------
#         st.markdown("<h3 class='section-header'>Export Results</h3>", unsafe_allow_html=True)
#         col1, col2 = st.columns(2)

#         with col1:
#             if st.button("📄 Export as Excel"):
#                 df_export = df.drop(columns=["Text"])
#                 df_export.to_excel("shortlist_results.xlsx", index=False)
#                 st.success("Saved as shortlist_results.xlsx")

#         with col2:
#             if st.button("📑 Export as JSON"):
#                 with open("shortlist_results.json", "w") as f:
#                     json.dump(results, f, indent=2)
#                 st.success("Saved as shortlist_results.json")

# # ------------------------------------------------------------
# # MANAGE DATABASES PAGE
# # ------------------------------------------------------------
# # ------------------------------------------------------------
# # MANAGE DATABASES PAGE
# # ------------------------------------------------------------
# def render_manage_databases_page(embeddings):
#     load_custom_css()

#     st.markdown("<h1 class='main-title'>📁 Manage Databases</h1>", unsafe_allow_html=True)

#     tabs = st.tabs(["📚 Historical Database", "⭐ Current Database"])

#     def render_db(db_key):
#         meta = load_metadata(db_key)

#         if not meta:
#             st.warning("Database is empty.")
#             return

#         st.markdown(f"<div class='card'>Total resumes stored: <b>{len(meta)}</b></div>", unsafe_allow_html=True)

#         df = pd.DataFrame.from_dict(meta, orient="index")

#         for col in ["original_name", "name", "surname", "email", "phone", "upload_date", "file_size"]:
#             if col not in df.columns:
#                 df[col] = "-"

#         view_df = df[["original_name", "name", "surname", "email", "phone", "upload_date", "file_size"]]
#         view_df.columns = ["File Name", "First Name", "Last Name", "Email", "Phone", "Upload Date", "Size"]

#         builder = GridOptionsBuilder.from_dataframe(view_df)
#         builder.configure_default_column(editable=False, filter=True, sortable=True)
#         builder.configure_pagination(enabled=True)
#         builder.configure_selection("multiple")

#         grid = AgGrid(view_df, gridOptions=builder.build(), height=380, theme="balham", key=f"{db_key}_grid")
#         selected = grid["selected_rows"]

#         st.markdown("<h3 class='section-header'>Delete Selected</h3>", unsafe_allow_html=True)

#         if st.button(f"🗑️ Delete from {db_key.title()}", key=f"delete_{db_key}"):
#             if not selected:
#                 st.warning("No resumes selected.")
#             else:
#                 for row in selected:
#                     for fid, m in meta.items():
#                         if m["original_name"] == row["File Name"]:
#                             delete_from_db(fid, db_key, embeddings)
#                 st.success("Deleted successfully.")
#                 st.rerun()

#         if db_key == "current":
#             st.markdown("<h3 class='section-header'>Archive</h3>", unsafe_allow_html=True)

#             if st.button("📦 Archive Current → Historical", key="archive_db"):
#                 hist_meta = load_metadata("historical")

#                 for fid, info in meta.items():
#                     hist_meta[fid] = info

#                 save_metadata(hist_meta, "historical")

#                 clear_database("current")

#                 st.success("Archived to Historical DB successfully.")
#                 st.rerun()

#         st.markdown("<h3 class='section-header'>Danger Zone</h3>", unsafe_allow_html=True)
#         if st.button(f"🔥 Clear Entire {db_key.title()} Database", key=f"clear_{db_key}"):
#             clear_database(db_key)
#             st.success(f"{db_key.title()} database cleared.")
#             st.rerun()

#     with tabs[0]:
#         render_db("historical")

#     with tabs[1]:
#         render_db("current")


# # ------------------------------------------------------------
# # SETTINGS PAGE
# # ------------------------------------------------------------
# def render_settings_page():
#     load_custom_css()

#     st.markdown("<h1 class='main-title'>⚙️ Application Settings</h1>", unsafe_allow_html=True)

#     st.markdown("<div class='card'>Modify preferences related to the AI model, UI theme, and scoring thresholds.</div>", unsafe_allow_html=True)

#     # ------------------ LLM Model Selection ------------------
#     st.markdown("<h3 class='section-header'>LLM Model</h3>", unsafe_allow_html=True)

#     model = st.selectbox(
#         "Choose AI Model",
#         ["llama-3.1-8b-instant", "llama-3.1-70b", "mixtral-8x7b", "gemma-7b"],
#         index=0
#     )

#     st.info(f"Selected model will apply after page restart: **{model}**")

#     # ------------------- Score Thresholds --------------------
#     st.markdown("<h3 class='section-header'>Shortlisting Thresholds</h3>", unsafe_allow_html=True)

#     shortlist_threshold = st.slider(
#         "Minimum Score for Shortlisting",
#         min_value=0,
#         max_value=100,
#         value=60
#     )

#     excellent_threshold = st.slider(
#         "Excellent Candidate Score",
#         min_value=0,
#         max_value=100,
#         value=75
#     )

#     st.success(f"Updated thresholds: Shortlist ≥ {shortlist_threshold}, Excellent ≥ {excellent_threshold}")

#     # ------------------- UI Theme Options --------------------
#     st.markdown("<h3 class='section-header'>Theme</h3>", unsafe_allow_html=True)

#     theme = st.selectbox(
#         "UI Theme",
#         ["Dark (recommended)", "Light", "High Contrast"]
#     )

#     st.info(f"Selected theme: **{theme}** (Will apply in next release)")

#     # ------------------- Save Button -------------------------
#     st.button("💾 Save Settings", type="primary")
# # ------------------------------------------------------------
# # ------------------------------------------------------------
# # FINAL MAIN APP CONTROLLER (MERGED + CLEANED)
# # ------------------------------------------------------------
# def main():

#     st.set_page_config(
#         page_title="Dual-DB Resume Shortlister",
#         page_icon="🤖",
#         layout="wide"
#     )

#     # Load custom CSS theme
#     load_custom_css()

#     # Initialize storage folders
#     init_storage()

#     # Load AI Models
#     llm = load_llm()
#     embeddings = load_embeddings()

#     # Sidebar logo + Navigation
#     menu = sidebar_logo()

#     # Page routing
#     if menu == "🏠 Dashboard":
#         render_dashboard_home ()

#     elif menu == "📤 Upload Resumes":
#        render_upload_page(embeddings)

#     elif menu == "📊 Shortlist Candidates":
#        render_shortlist_page(llm, embeddings)

#     elif menu == "📁 Manage Databases":
#       render_manage_databases_page(embeddings)

#     elif menu == "⚙️ Settings":
#       render_settings_page()



# # ------------------------------------------------------------
# # RUN APPLICATION
# # ------------------------------------------------------------
# if __name__ == "__main__":
#     main()

















# import os
# import json
# import tempfile
# import shutil
# import base64
# import streamlit as st
# import pandas as pd
# from dotenv import load_dotenv
# from pathlib import Path
# from datetime import datetime
# import plotly.express as px
# from st_aggrid import AgGrid, GridOptionsBuilder

# # LangChain Imports
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.documents import Document

# # Load environment variables
# load_dotenv()

# # ------------------------------------------------------------
# # DATABASE CONFIG
# # ------------------------------------------------------------
# BASE_DB_DIR = "vector_db"
# DB_TYPES = {
#     "historical": {
#         "path": os.path.join(BASE_DB_DIR, "historical"),
#         "meta": os.path.join(BASE_DB_DIR, "historical", "metadata.json")
#     },
#     "current": {
#         "path": os.path.join(BASE_DB_DIR, "current"),
#         "meta": os.path.join(BASE_DB_DIR, "current", "metadata.json")
#     }
# }

# # ------------------------------------------------------------
# # INITIALIZE STORAGE DIRECTORIES
# # ------------------------------------------------------------
# def init_storage():
#     for db_key, cfg in DB_TYPES.items():
#         Path(cfg["path"]).mkdir(parents=True, exist_ok=True)
#         if not os.path.exists(cfg["meta"]):
#             with open(cfg["meta"], "w") as f:
#                 json.dump({}, f)

# # ------------------------------------------------------------
# # EMBEDDING LOADER (CACHED)
# # ------------------------------------------------------------
# @st.cache_resource
# def load_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )

# # ------------------------------------------------------------
# # METADATA OPERATIONS
# # ------------------------------------------------------------
# def load_metadata(db_type):
#     try:
#         with open(DB_TYPES[db_type]["meta"], "r") as f:
#             return json.load(f)
#     except:
#         return {}

# def save_metadata(meta, db_type):
#     with open(DB_TYPES[db_type]["meta"], "w") as f:
#         json.dump(meta, f, indent=2)

# # ------------------------------------------------------------
# # VECTORSTORE LOADING & SAVING
# # ------------------------------------------------------------
# def get_vectorstore(embeddings, db_type):
#     index_path = os.path.join(DB_TYPES[db_type]["path"], "faiss_index")
    
#     if os.path.exists(index_path):
#         try:
#             return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
#         except:
#             return None
#     return None

# def save_vectorstore(vs, db_type):
#     index_path = os.path.join(DB_TYPES[db_type]["path"], "faiss_index")
#     vs.save_local(index_path)

# # ------------------------------------------------------------
# # ADD DOCUMENTS TO VECTORSTORE
# # ------------------------------------------------------------
# def add_to_vectordb(docs, embeddings, db_type, file_id, original_name, meta_entry):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = splitter.split_documents(docs)

#     for s in splits:
#         s.metadata.update({
#             "resume_id": file_id,
#             "original_name": original_name,
#             "db_type": db_type
#         })

#     vs = get_vectorstore(embeddings, db_type)

#     if vs is None:
#         vs = FAISS.from_documents(splits, embeddings)
#     else:
#         vs.add_documents(splits)

#     save_vectorstore(vs, db_type)

#     meta = load_metadata(db_type)
#     meta[file_id] = meta_entry
#     save_metadata(meta, db_type)

# # ------------------------------------------------------------
# # PROCESS FILE UPLOAD
# # ------------------------------------------------------------
# def process_upload(uploaded_file, embeddings, target_db="historical"):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_id = f"{timestamp}_{uploaded_file.name}"

#     ext = uploaded_file.name.split(".")[-1].lower()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
#         tmp.write(uploaded_file.read())
#         tmp_path = tmp.name

#     try:
#         loader = PyPDFLoader(tmp_path) if ext == "pdf" else TextLoader(tmp_path, encoding="utf-8")
#         docs = loader.load()

#         full_text = "\n".join([d.page_content for d in docs])

#         meta_entry = {
#             "original_name": uploaded_file.name,
#             "upload_date": timestamp,
#             "file_size": uploaded_file.size,
#             "preview": full_text[:300],
#             "name": "-",
#             "surname": "-",
#             "phone": "-",
#             "email": "-"
#         }

#         add_to_vectordb(docs, embeddings, target_db, file_id, uploaded_file.name, meta_entry)

#         return file_id, docs, full_text

#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

# # ------------------------------------------------------------
# # RETRIEVE RESUME TEXT FROM VECTORSTORE
# # ------------------------------------------------------------
# def get_resume_text(file_id, db_type, embeddings):
#     vs = get_vectorstore(embeddings, db_type)
#     if not vs:
#         return None

#     all_docs = vs.similarity_search("", k=5000)
#     docs = [d for d in all_docs if d.metadata.get("resume_id") == file_id]

#     if not docs:
#         return None

#     return "\n\n".join([d.page_content for d in docs])

# # ------------------------------------------------------------
# # DELETE RESUME FROM DB
# # ------------------------------------------------------------
# def delete_from_db(file_id, db_type, embeddings):
#     meta = load_metadata(db_type)
#     if file_id in meta:
#         del meta[file_id]
#         save_metadata(meta, db_type)

#     vs = get_vectorstore(embeddings, db_type)
#     if not vs:
#         return

#     all_docs = vs.similarity_search("", k=8000)
#     filtered = [d for d in all_docs if d.metadata.get("resume_id") != file_id]

#     if not filtered:
#         shutil.rmtree(os.path.join(DB_TYPES[db_type]["path"], "faiss_index"), ignore_errors=True)
#     else:
#         new_vs = FAISS.from_documents(filtered, embeddings)
#         save_vectorstore(new_vs, db_type)

# # ------------------------------------------------------------
# # CLEAR ENTIRE DATABASE
# # ------------------------------------------------------------
# def clear_database(db_type):
#     shutil.rmtree(DB_TYPES[db_type]["path"], ignore_errors=True)
#     init_storage()

# # ------------------------------------------------------------
# # LOAD LLM
# # ------------------------------------------------------------
# @st.cache_resource
# def load_llm():
#     return ChatGroq(
#         model_name="llama-3.1-8b-instant",
#         api_key=os.getenv("GROQ_API_KEY"),
#         temperature=0
#     )

# # ------------------------------------------------------------
# # SHORTLIST EVALUATION LOGIC
# # ------------------------------------------------------------
# SHORTLIST_PROMPT = """
# You are an expert HR AI. Evaluate the resume against the job description.

# Rules:
# 1. Score from 0-100.
# 2. If score >= 60 → Shortlisted else Rejected.
# 3. Extract: name, surname, email, phone.
# 4. Return ONLY JSON.

# Format:
# {{
#   "name": "",
#   "surname": "",
#   "email": "",
#   "phone": "",
#   "score": 0,
#   "decision": "",
#   "reason": ""
# }}

# RESUME:
# {resume_text}

# JOB DESCRIPTION:
# {jd}
# """

# def evaluate_candidate(llm, resume_text, jd):
#     prompt = ChatPromptTemplate.from_template(SHORTLIST_PROMPT)
#     chain = prompt | llm | StrOutputParser()

#     try:
#         response = chain.invoke({"resume_text": resume_text, "jd": jd})
#         if not response or "{" not in response:
#             return None
        
#         start = response.find("{")
#         end = response.rfind("}") + 1
#         cleaned = response[start:end]

#         return json.loads(cleaned)

#     except Exception as e:
#         st.error(f"LLM error: {e}")
#         return None

# # ------------------------------------------------------------
# # CUSTOM CSS STYLING
# # ------------------------------------------------------------
# def load_custom_css():
#     st.markdown("""
#     <style>

#     /* Sidebar Container */
#     [data-testid="stSidebar"] {
#         background-color: #E8F1FA !important;  
#         padding-top: 30px !important;
#         border-right: 1px solid #CFD9E6 !important; 
#     }

#     /* Logo inside sidebar */
#     .sidebar-logo img {
#         width: 120px !important;
#         border-radius: 50% !important;
#         display: block !important;
#         margin-left: auto !important;
#         margin-right: auto !important;
#         margin-top: 10px !important;
#         margin-bottom: 25px !important;
#     }

#     /* Navigation Title */
#     .sidebar-title {
#         color: #1E3A8A !important;  
#         font-size: 20px !important;
#         font-weight: 700 !important;
#         padding-left: 20px !important;
#         margin-bottom: 20px !important;
#     }

#     /* Navigation Radio Buttons Text */
#     .sidebar .stRadio label {
#         color: #1F2937 !important;  
#         font-size: 16px !important;
#         font-weight: 600 !important;
#         padding: 5px 10px !important;
#         border-radius: 6px !important;
#     }

#     /* Hover Effect */
#     [data-testid="stSidebar"] .stRadio label:hover {
#         background-color: #DCEBFA !important;
#         color: #1E40AF !important;
#     }

#     /* Active Selected Option */
#     [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-selected="true"] {
#         background-color: #D0E4FF !important;
#         color: #1E3A8A !important;
#         font-weight: 700 !important;
#         border-radius: 6px !important;
#         padding: 6px 12px !important;
#     }

#     /* Score Badges */
#     .badge-pass { background-color: #DEF7EC; color: #03543F; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
#     .badge-mid { background-color: #FEF3C7; color: #92400E; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
#     .badge-low { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 8px; border-radius: 4px; font-weight: bold; }

#     /* Cards */
#     .card {
#         padding: 20px;
#         background-color: #FFFFFF;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         margin-bottom: 20px;
#         color: #333;
#     }
#     .main-title {
#         color: #1E3A8A;
#         font-weight: 800;
#         margin-bottom: 10px;
#     }
#     .section-header {
#         color: #1E40AF;
#         margin-top: 20px;
#         margin-bottom: 10px;
#         border-bottom: 2px solid #E5E7EB;
#         padding-bottom: 5px;
#     }

#     </style>
#     """, unsafe_allow_html=True)

# # ------------------------------------------------------------
# # SIDEBAR NAVIGATION
# # ------------------------------------------------------------
# def sidebar_logo():
#     # Make sure 'nx.png' exists or replace with a placeholder logic
#     logo_path = "nx.png"
    
#     if os.path.exists(logo_path):
#         logo_encoded = base64.b64encode(open(logo_path, "rb").read()).decode()
#         st.sidebar.markdown(
#             f"""
#             <div class="sidebar-logo">
#                 <img src="data:image/png;base64,{logo_encoded}">
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#     else:
#         st.sidebar.markdown("<h2 style='text-align:center;'>HR AI</h2>", unsafe_allow_html=True)

#     st.sidebar.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)

#     return st.sidebar.radio(
#         "",
#         [
#             "🏠 Dashboard",
#             "📤 Upload Resumes",
#             "📊 Shortlist Candidates",
#             "📁 Manage Databases",
#             "⚙️ Settings"
#         ]
#     )

# # ------------------------------------------------------------
# # PAGE 1: DASHBOARD HOME
# # ------------------------------------------------------------
# def render_dashboard_home():
#     load_custom_css()

#     st.markdown("<h1 class='main-title'>HR Analytics Dashboard</h1>", unsafe_allow_html=True)
#     st.markdown("<div class='card'>A complete overview of your hiring pipeline and screening activity.</div>", unsafe_allow_html=True)

#     # Load metadata
#     hist_meta = load_metadata("historical")
#     curr_meta = load_metadata("current")

#     # Summary metrics
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Resumes", len(hist_meta))
#     with col2:
#         st.metric("Shortlisted (Current DB)", len(curr_meta))
#     with col3:
#         today = datetime.now().strftime("%Y%m%d")
#         today_count = sum(1 for v in hist_meta.values() if v["upload_date"].startswith(today))
#         st.metric("Uploaded Today", today_count)

#     # Pie chart for shortlisted vs rejected
#     shortlisted = len(curr_meta)
#     rejected = max(0, len(hist_meta) - shortlisted)

#     fig = px.pie(
#         names=["Shortlisted", "Rejected"],
#         values=[shortlisted, rejected],
#         title="Shortlisting Distribution"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # Upload trends chart
#     upload_dates = [
#         v["upload_date"][:8]  # extract YYYYMMDD
#         for v in hist_meta.values()
#     ]

#     if upload_dates:
#         df = pd.DataFrame(upload_dates, columns=["date"])
#         upload_trend = df["date"].value_counts().sort_index()

#         fig2 = px.line(
#             x=upload_trend.index,
#             y=upload_trend.values,
#             title="Resume Upload Trend"
#         )
#         fig2.update_layout(xaxis_title="Date", yaxis_title="Uploads")
#         st.plotly_chart(fig2, use_container_width=True)

# # ------------------------------------------------------------
# # PAGE 2: UPLOAD RESUMES
# # ------------------------------------------------------------
# def render_upload_page(embeddings):
#     load_custom_css()

#     st.markdown("<h1 class='main-title'>📤 Upload Resumes</h1>", unsafe_allow_html=True)
#     st.markdown(
#         "<div class='card'>Upload resumes into the CURRENT database. "
#         "After shortlisting, HR can archive them into the Historical Database.</div>",
#         unsafe_allow_html=True
#     )

#     uploaded_files = st.file_uploader(
#         "Drag & drop PDF/TXT files",
#         type=["pdf", "txt"],
#         accept_multiple_files=True,
#         key="resume_uploader"
#     )

#     if st.button("Process Uploads", type="primary", key="process_upload_button"):
#         if not uploaded_files:
#             st.warning("Please upload at least one resume.")
#             return

#         # Check duplicates only in CURRENT DB
#         existing_meta = load_metadata("current")
#         existing_names = {m["original_name"].lower() for m in existing_meta.values()}

#         duplicates = []
#         new_files = []

#         for f in uploaded_files:
#             if f.name.lower() in existing_names:
#                 duplicates.append(f.name)
#             else:
#                 new_files.append(f)

#         if duplicates:
#             st.warning("⚠️ Duplicate resumes skipped:\n" + "\n".join([f"- {x}" for x in duplicates]))

#         if not new_files:
#             st.error("All uploaded resumes were duplicates in CURRENT DB.")
#             return

#         progress = st.progress(0)
#         uploaded_info = []

#         for idx, f in enumerate(new_files):
#             try:
#                 file_id, docs, text = process_upload(f, embeddings, "current")
#                 uploaded_info.append({
#                     "File Name": f.name,
#                     "Preview": text[:200] + "..."
#                 })
#             except Exception as e:
#                 st.error(f"Error processing {f.name}: {e}")

#             progress.progress((idx + 1) / len(new_files))

#         st.success("🎉 Upload complete! Added to CURRENT database.")

#         if uploaded_info:
#             st.markdown("<h3 class='section-header'>Uploaded Resumes Summary</h3>", unsafe_allow_html=True)
#             df = pd.DataFrame(uploaded_info)
#             builder = GridOptionsBuilder.from_dataframe(df)
#             builder.configure_default_column(editable=False)
#             builder.configure_pagination(enabled=True)
#             AgGrid(df, gridOptions=builder.build(), height=300, theme="balham", key="upload_summary_grid")

# # ------------------------------------------------------------
# # PAGE 3: AI SHORTLISTING (UPDATED WITH DELETE & CSV)
# # ------------------------------------------------------------
# def render_shortlist_page(llm, embeddings):
#     load_custom_css()

#     st.markdown("<h1 class='main-title'>📊 AI Resume Shortlisting</h1>", unsafe_allow_html=True)
#     st.markdown("""
#         <div class='card'>
#             The AI evaluates each resume against the provided Job Description and generates a score,
#             decision, and reasoning. HR can export the report and later archive the Current Database.
#         </div>
#     """, unsafe_allow_html=True)

#     # JD Templates
#     st.markdown("<h3 class='section-header'>Job Description</h3>", unsafe_allow_html=True)
#     template_option = st.selectbox(
#         "Select a JD Template",
#         [
#             "Write JD manually",
#             "Software Engineer",
#             "Data Analyst",
#             "Backend Developer",
#             "Flutter Developer",
#             "Product Manager",
#             "DevOps Engineer"
#         ]
#     )

#     jd_presets = {
#         "Software Engineer": "We need a Software Engineer skilled in DSA, Python/Java, OOP, Git, SQL, API development, and SDLC.",
#         "Data Analyst": "We seek a Data Analyst skilled in Excel, SQL, Python, dashboards, visualizations, and reporting.",
#         "Backend Developer": "We need a Backend Engineer skilled in Node.js, Express, REST APIs, MongoDB/MySQL, and authentication.",
#         "Flutter Developer": "We need a Flutter Developer with Dart skills, REST API integration, state management, and UI/UX.",
#         "Product Manager": "We need a PM experienced in PRDs, roadmaps, KPIs, user interviews, and cross-team communication.",
#         "DevOps Engineer": "We seek a DevOps Engineer skilled in AWS, Docker, CI/CD pipelines, Kubernetes, and monitoring."
#     }

#     if template_option == "Write JD manually":
#         jd = st.text_area("Enter JD", height=150)
#     else:
#         jd = st.text_area("Enter JD", jd_presets[template_option], height=150)

#     # Candidate Source Info
#     st.markdown("<h3 class='section-header'>Candidate Source</h3>", unsafe_allow_html=True)
#     st.info("Shortlisting will be performed on **Current Database** only.")
    
#     db_key = "current"
#     metadata = load_metadata("current")
#     st.info(f"Total candidates in Current Database: **{len(metadata)}**")

#     # Start Evaluation
#     if st.button("🚀 Run AI Evaluation", type="primary"):
#         if not jd:
#             st.warning("Please enter a Job Description.")
#             return
#         if not metadata:
#             st.error("No candidates available in Current Database.")
#             return

#         progress = st.progress(0)
#         results = []
#         total = len(metadata)

#         for idx, (file_id, info) in enumerate(metadata.items()):
#             resume_text = get_resume_text(file_id, db_key, embeddings)
#             if not resume_text:
#                 continue

#             eval_result = evaluate_candidate(llm, resume_text, jd)
#             if not eval_result:
#                 continue

#             score = eval_result.get("score", 0)

#             if score >= 75:
#                 badge = "<span class='badge-pass'>Excellent</span>"
#             elif score >= 60:
#                 badge = "<span class='badge-mid'>Good</span>"
#             else:
#                 badge = "<span class='badge-low'>Poor</span>"

#             # Update Metadata
#             info["name"] = eval_result.get("name", "-")
#             info["surname"] = eval_result.get("surname", "-")
#             info["phone"] = eval_result.get("phone", "-")
#             info["email"] = eval_result.get("email", "-")
#             save_metadata(metadata, db_key)

#             results.append({
#                 "Name": info["name"],
#                 "Surname": info["surname"],
#                 "Email": info["email"],
#                 "Phone": info["phone"],
#                 "Score": score,
#                 "Badge": badge,
#                 "Decision": eval_result.get("decision", ""),
#                 "Reason": eval_result.get("reason", ""),
#                 "File Name": info["original_name"],
#                 "Resume ID": file_id,
#                 "Promoted": "No",
#                 "Text": resume_text
#             })

#             progress.progress((idx + 1) / total)

#         if not results:
#             st.error("No results generated.")
#             return

#         st.success("🎉 Evaluation Complete!")
#         # Save to session state so it persists
#         st.session_state['shortlist_results'] = results

#     # Display Results if they exist in Session State
#     if 'shortlist_results' in st.session_state:
#         results = st.session_state['shortlist_results']
#         df = pd.DataFrame(results)

#         st.markdown("<h3 class='section-header'>Evaluation Results</h3>", unsafe_allow_html=True)

#         builder = GridOptionsBuilder.from_dataframe(df)
#         builder.configure_pagination(enabled=True)
#         builder.configure_default_column(editable=False, filter=True, sortable=True)
#         builder.configure_column("Badge", cellRenderer="html")
#         builder.configure_selection("single")

#         grid = AgGrid(
#             df,
#             gridOptions=builder.build(),
#             height=420,
#             theme="material",
#             update_mode="MODEL_CHANGED",
#             key="shortlist_grid"
#         )

#         selected = grid["selected_rows"]

#         # Resume Viewer & Delete Functionality
#         if selected:
#             st.markdown("<h3 class='section-header'>Resume Viewer & Actions</h3>", unsafe_allow_html=True)
#             row = selected[0]

#             col_view, col_action = st.columns([3, 1])

#             with col_view:
#                 st.write(f"### {row['Name']} {row['Surname']}")
#                 st.write(f"📧 {row['Email']}")
#                 st.write(f"📞 {row['Phone']}")
#                 st.markdown(row["Badge"], unsafe_allow_html=True)
            
#             with col_action:
#                 st.write("Does this candidate not fit?")
#                 if st.button("🗑️ Delete Candidate", type="secondary"):
#                     delete_from_db(row["Resume ID"], "current", embeddings)
#                     st.warning(f"Deleted {row['Name']} {row['Surname']}. Please re-run evaluation or refresh.")
                    
#                     # Remove from current session results
#                     st.session_state['shortlist_results'] = [r for r in results if r['Resume ID'] != row['Resume ID']]
#                     st.rerun()

#             st.write("---")
#             st.text_area("Resume Text", row["Text"], height=350)

#         # Export Buttons (Excel, JSON, CSV)
#         st.markdown("<h3 class='section-header'>Export Results</h3>", unsafe_allow_html=True)
#         col1, col2, col3 = st.columns(3)

#         df_export = df.drop(columns=["Text", "Badge"]) # Clean dataframe for export

#         with col1:
#             if st.button("📄 Export as Excel"):
#                 df_export.to_excel("shortlist_results.xlsx", index=False)
#                 st.success("Saved as shortlist_results.xlsx")

#         with col2:
#             if st.button("📑 Export as JSON"):
#                 with open("shortlist_results.json", "w") as f:
#                     json.dump(results, f, indent=2)
#                 st.success("Saved as shortlist_results.json")

#         with col3:
#             csv_data = df_export.to_csv(index=False).encode('utf-8')
#             st.download_button(
#                 label="📄 Download CSV",
#                 data=csv_data,
#                 file_name="shortlist_results.csv",
#                 mime="text/csv",
#                 key="download-csv"
#             )

# # ------------------------------------------------------------
# # PAGE 4: MANAGE DATABASES
# # ------------------------------------------------------------
# def render_manage_databases_page(embeddings):
#     load_custom_css()
#     st.markdown("<h1 class='main-title'>📁 Manage Databases</h1>", unsafe_allow_html=True)

#     tabs = st.tabs(["📚 Historical Database", "⭐ Current Database"])

#     def render_db(db_key):
#         meta = load_metadata(db_key)

#         if not meta:
#             st.warning("Database is empty.")
#             return

#         st.markdown(f"<div class='card'>Total resumes stored: <b>{len(meta)}</b></div>", unsafe_allow_html=True)

#         df = pd.DataFrame.from_dict(meta, orient="index")
#         for col in ["original_name", "name", "surname", "email", "phone", "upload_date", "file_size"]:
#             if col not in df.columns:
#                 df[col] = "-"

#         view_df = df[["original_name", "name", "surname", "email", "phone", "upload_date", "file_size"]]
#         view_df.columns = ["File Name", "First Name", "Last Name", "Email", "Phone", "Upload Date", "Size"]

#         builder = GridOptionsBuilder.from_dataframe(view_df)
#         builder.configure_default_column(editable=False, filter=True, sortable=True)
#         builder.configure_pagination(enabled=True)
#         builder.configure_selection("multiple")

#         grid = AgGrid(view_df, gridOptions=builder.build(), height=380, theme="balham", key=f"{db_key}_grid")
#         selected = grid["selected_rows"]

#         st.markdown("<h3 class='section-header'>Delete Selected</h3>", unsafe_allow_html=True)
#         if st.button(f"🗑️ Delete from {db_key.title()}", key=f"delete_{db_key}"):
#             if not selected:
#                 st.warning("No resumes selected.")
#             else:
#                 for row in selected:
#                     # Reverse lookup file_id
#                     for fid, m in meta.items():
#                         if m["original_name"] == row["File Name"]:
#                             delete_from_db(fid, db_key, embeddings)
#                 st.success("Deleted successfully.")
#                 st.rerun()

#         if db_key == "current":
#             st.markdown("<h3 class='section-header'>Archive</h3>", unsafe_allow_html=True)
#             if st.button("📦 Archive Current → Historical", key="archive_db"):
#                 hist_meta = load_metadata("historical")
#                 for fid, info in meta.items():
#                     hist_meta[fid] = info
#                 save_metadata(hist_meta, "historical")
#                 clear_database("current")
#                 st.success("Archived to Historical DB successfully.")
#                 st.rerun()

#         st.markdown("<h3 class='section-header'>Danger Zone</h3>", unsafe_allow_html=True)
#         if st.button(f"🔥 Clear Entire {db_key.title()} Database", key=f"clear_{db_key}"):
#             clear_database(db_key)
#             st.success(f"{db_key.title()} database cleared.")
#             st.rerun()

#     with tabs[0]:
#         render_db("historical")
#     with tabs[1]:
#         render_db("current")

# # ------------------------------------------------------------
# # PAGE 5: SETTINGS
# # ------------------------------------------------------------
# def render_settings_page():
#     load_custom_css()
#     st.markdown("<h1 class='main-title'>⚙️ Application Settings</h1>", unsafe_allow_html=True)
#     st.markdown("<div class='card'>Modify preferences related to the AI model, UI theme, and scoring thresholds.</div>", unsafe_allow_html=True)

#     st.markdown("<h3 class='section-header'>LLM Model</h3>", unsafe_allow_html=True)
#     model = st.selectbox(
#         "Choose AI Model",
#         ["llama-3.1-8b-instant", "llama-3.1-70b", "mixtral-8x7b", "gemma-7b"],
#         index=0
#     )
#     st.info(f"Selected model will apply after page restart: **{model}**")

#     st.markdown("<h3 class='section-header'>Shortlisting Thresholds</h3>", unsafe_allow_html=True)
#     shortlist_threshold = st.slider("Minimum Score for Shortlisting", 0, 100, 60)
#     excellent_threshold = st.slider("Excellent Candidate Score", 0, 100, 75)
#     st.success(f"Updated thresholds: Shortlist ≥ {shortlist_threshold}, Excellent ≥ {excellent_threshold}")

#     st.markdown("<h3 class='section-header'>Theme</h3>", unsafe_allow_html=True)
#     theme = st.selectbox("UI Theme", ["Dark (recommended)", "Light", "High Contrast"])
#     st.info(f"Selected theme: **{theme}** (Will apply in next release)")

#     st.button("💾 Save Settings", type="primary")

# # ------------------------------------------------------------
# # MAIN CONTROLLER
# # ------------------------------------------------------------
# def main():
#     st.set_page_config(
#         page_title="Dual-DB Resume Shortlister",
#         page_icon="🤖",
#         layout="wide"
#     )

#     load_custom_css()
#     init_storage()
    
#     # Load Models
#     llm = load_llm()
#     embeddings = load_embeddings()

#     # Sidebar & Routing
#     menu = sidebar_logo()

#     if menu == "🏠 Dashboard":
#         render_dashboard_home()
#     elif menu == "📤 Upload Resumes":
#         render_upload_page(embeddings)
#     elif menu == "📊 Shortlist Candidates":
#         render_shortlist_page(llm, embeddings)
#     elif menu == "📁 Manage Databases":
#         render_manage_databases_page(embeddings)
#     elif menu == "⚙️ Settings":
#         render_settings_page()

# if __name__ == "__main__":

#     main()









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
from pinecone import Pinecone
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
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Ensure Index Exists
existing_indexes = [i.name for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    try:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=serverlessSpec(cloud="aws", region="us-east-1")
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






