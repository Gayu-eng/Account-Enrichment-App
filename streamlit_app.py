import streamlit as st
import pandas as pd
import json, io, os, requests, datetime, time
from concurrent.futures import ThreadPoolExecutor
from serpapi.google_search import GoogleSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Securely fetch keys from Streamlit Cloud Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]
SLACK_BOT_TOKEN = st.secrets["SLACK_BOT_TOKEN"]

st.set_page_config(page_title="Enterprise Account Research", layout="wide")

if "page" not in st.session_state: st.session_state.page = "home"
if "refresh_keys" not in st.session_state: st.session_state.refresh_keys = {}

with st.sidebar:
    st.title("⚙️ System Control")
    enrichment_mode = st.radio("Enrichment Mode", ["Real Data", "Test Data"])
    st.divider()
    target_refresh = st.text_input("Company to Refresh")
    if st.button("♻️ Invalidate Specific Company"):
        if target_refresh:
            st.session_state.refresh_keys[target_refresh.strip()] = time.time()
            st.success(f"Key updated for {target_refresh}!")
    if st.button("🔄 Global Clear Cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.clear()
        st.session_state.page = "home"
        st.rerun()

@st.cache_resource(show_spinner=False)
def load_models():
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0)
    emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return llm, emb

def fetch_deep_results(query, mode):
    if mode == "Test Data": return ["DUMMY DATA MODE"]
    search = GoogleSearch({"q": query, "api_key": SERPAPI_API_KEY, "num": 10})
    res = search.get_dict()
    return [f"Source: {r.get('link')} | Content: {r.get('snippet')}" for r in res.get('organic_results', [])]

@st.cache_data(show_spinner="Searching...", persist="disk", max_entries=1000, ttl=86400)
def run_deep_rag(company, context, _llm, _emb, cache_key=None):
    if context == "DUMMY DATA MODE":
        return {"website": "example.com", "about": "Dummy info", "tech_stack": "N/A"}
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents([Document(page_content=context)])
    vectorstore = FAISS.from_documents(docs, _emb)
    relevant = vectorstore.similarity_search(company, k=15)
    subset = "\n".join([d.page_content for d in relevant])
    
    res = _llm.invoke([SystemMessage(content="Return JSON"), HumanMessage(content=f"Extract data for {company} from {subset}")])
    try: return json.loads(res.content.replace('```json', '').replace('```', '').strip())
    except: return {}

if st.session_state.page == "home":
    st.title("Account Enrichment App")
    comps = st.text_area("Company Names:")
    if st.button("Run Research"):
        llm, emb = load_models()
        results = []
        for c in [x.strip() for x in comps.split("\n") if x.strip()]:
            c_key = st.session_state.refresh_keys.get(c, "v1")
            snippets = fetch_deep_results(f"{c} headquarters", enrichment_mode)
            data = run_deep_rag(c, " ".join(snippets), llm, emb, cache_key=c_key)
            results.append({"Company": c, **data})
        st.session_state.df = pd.DataFrame(results)
        st.rerun()

    if "df" in st.session_state:
        st.dataframe(st.session_state.df)
        if st.button("💬 Slack DM"): 
            st.session_state.page = "slack"
            st.rerun()

elif st.session_state.page == "slack":
    st.title("Slack Configuration")
    uid = st.text_input("Slack ID")
    if st.button("Send"):
        st.success("Simulated send (check secrets configuration)")
    if st.button("Back"): 
        st.session_state.page = "home"
        st.rerun()
