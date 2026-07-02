
import streamlit as st
import pandas as pd
import json, io, os, requests, datetime
from concurrent.futures import ThreadPoolExecutor
from serpapi.google_search import GoogleSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

st.set_page_config(page_title="Enterprise Account Research", layout="wide")

# Securely load keys from Streamlit Secrets
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SLACK_TOKEN = st.secrets["SLACK_BOT_TOKEN"]

if "page" not in st.session_state: st.session_state.page = "home"

@st.cache_resource
def load_models():
    return ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0), OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def fetch_deep_results(query):
    search = GoogleSearch({"q": query, "api_key": SERPAPI_KEY, "num": 10})
    res = search.get_dict()
    return [f"Source: {r.get('link')} | Content: {r.get('snippet')}" for r in res.get('organic_results', [])]

@st.cache_data(show_spinner=False)
def run_deep_rag(company, context, _llm, _emb):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents([Document(page_content=context)])
    vectorstore = FAISS.from_documents(docs, _emb)
    relevant = vectorstore.similarity_search(company, k=15)
    subset = "\n".join([d.page_content for d in relevant])
    
    system_msg = SystemMessage(content="You are a professional corporate investigator. Return ONLY raw JSON.")
    prompt = f"Extract fields for {company}: website, about, city, state, country, zip, email, phone, linkedin_url, article_1, article_2, article_3, tech_stack, funding_stage, total_funding, last_funding_date, lead_investors.\n\nContext: {subset}"
    res = _llm.invoke([system_msg, HumanMessage(content=prompt)])
    try:
        return json.loads(res.content.replace('```json', '').replace('```', '').strip())
    except: return {}

# --- UI LOGIC ---
if st.session_state.page == "home":
    st.title("Ἶ2 Deep Account Enrichment")
    comps = st.text_area("Enter Company Names:", height=150)

    if st.button("Run Enrichment"):
        llm, emb = load_models()
        results = []
        comp_list = [x.strip() for x in comps.split("\n") if x.strip()]
        for c in comp_list:
            st.write(f"🔍 Researching: **{c}**...")
            queries = [f"{c} headquarters", f"{c} tech stack", f"{c} funding history"]
            snippets = []
            with ThreadPoolExecutor(max_workers=5) as ex:
                for r in ex.map(fetch_deep_results, queries): snippets.extend(r)
            data = run_deep_rag(c, " | ".join(snippets), llm, emb)
            results.append({"Company": c, **data})
        st.session_state.df = pd.DataFrame(results)
        st.rerun()

    if "df" in st.session_state:
        st.dataframe(st.session_state.df)
        if st.button("💬 Send to Slack DM"): 
            st.session_state.page = "slack"
            st.rerun()

elif st.session_state.page == "slack":
    st.title("💬 Slack Delivery")
    uid = st.text_input("Your Member ID")
    if st.button("🚀 Send Report"):
        headers = {"Authorization": f"Bearer {SLACK_TOKEN}"}
        res_dm = requests.post("https://slack.com/api/conversations.open", headers=headers, json={"users": uid}).json()
        if res_dm.get("ok"):
            target_id = res_dm['channel']['id']
            out = io.BytesIO(); st.session_state.df.to_excel(out, index=False)
            f_bytes = out.getvalue()
            r1 = requests.post("https://slack.com/api/files.getUploadURLExternal", headers=headers, data={"filename": "Report.xlsx", "length": len(f_bytes)}).json()
            if r1.get("ok"):
                requests.post(r1['upload_url'], data=f_bytes)
                requests.post("https://slack.com/api/files.completeUploadExternal", headers=headers, json={"files": [{"id": r1['file_id']}], "channel_id": target_id, "initial_comment": "Here is your report."})
                st.success("✅ Sent!")
    if st.button("Back"): 
        st.session_state.page = "home"
        st.rerun()
