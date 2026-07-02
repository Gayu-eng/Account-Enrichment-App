
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
if "page" not in st.session_state: st.session_state.page = "home"

# Hardcoded common configurations
SLACK_TOKEN = "xoxb-11024348506737-11486022280759-8DP7ycoABi40jmxtUXYm5iwO"

@st.cache_resource
def load_models():
    return ChatOpenAI(model="gpt-4o", openai_api_key="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", temperature=0), OpenAIEmbeddings(openai_api_key="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")

def fetch_deep_results(query):
    search = GoogleSearch({"q": query, "api_key": "96a5114d521fed441444d811bfda6aaea5a73fc2268be0c3eb788099dcae19cb", "num": 10})
    res = search.get_dict()
    snippets = [f"Source: {r.get('link')} | Content: {r.get('snippet')}" for r in res.get('organic_results', [])]
    return snippets

@st.cache_data(show_spinner=False)
def run_deep_rag(company, context, _llm, _emb):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents([Document(page_content=context)])
    vectorstore = FAISS.from_documents(docs, _emb)
    relevant = vectorstore.similarity_search(company, k=15)
    subset = "\n".join([d.page_content for d in relevant])

    system_msg = SystemMessage(content="You are a professional corporate investigator. Return ONLY raw JSON.")
    prompt = f"""Extract fields for {company}: website, about, city, state, country, zip, email, phone, linkedin_url, article_1, article_2, article_3, tech_stack, funding_stage, total_funding, last_funding_date, lead_investors.\n\nContext: {subset}"""
    res = _llm.invoke([system_msg, HumanMessage(content=prompt)])
    try:
        return json.loads(res.content.replace('```json', '').replace('```', '').strip())
    except:
        return {}

if st.session_state.page == "home":
    st.title("Ἶ2 Deep Account Enrichment")
    st.info("✅ **Funding Analysis Active:** Research now includes Funding Stage, Total Capital, and Lead Investors.")

    comps = st.text_area("Enter Company Names (One per line):", height=150, placeholder="Apple\nNvidia\nOpenAI")

    if st.button("Run Deep Enrichment"):
        if not comps.strip(): st.warning("⚠️ Please enter company names.")
        else:
            llm, emb = load_models()
            results = []
            comp_list = [x.strip() for x in comps.split("\n") if x.strip()]
            now = datetime.datetime.now()
            current_month_year = now.strftime("%B %Y")
            next_year = now.year + 1
            for c in comp_list:
                st.write(f"🔍 Researching: **{c}**...")
                queries = [
                    f"{c} corporate headquarters address",
                    f"{c} tech stack stackshare",
                    f"{c} crunchbase funding history",
                    f"{c} latest news articles {current_month_year}",
                    f"{c} news {next_year}"
                ]
                snippets = []
                with ThreadPoolExecutor(max_workers=5) as ex:
                    for r in ex.map(fetch_deep_results, queries): snippets.extend(r)
                data = run_deep_rag(c, " | ".join(snippets), llm, emb)
                results.append({"Company": c, **data})

            cols = ['Company', 'website', 'about', 'city', 'state', 'country', 'zip', 'email', 'phone', 'linkedin_url', 'article_1', 'article_2', 'article_3', 'tech_stack', 'funding_stage', 'total_funding', 'last_funding_date', 'lead_investors']
            st.session_state.df = pd.DataFrame(results).reindex(columns=cols)
            st.rerun()

    if "df" in st.session_state:
        st.subheader("Enrichment Results")
        st.dataframe(st.session_state.df)
        c1, c2 = st.columns(2)
        with c1:
            out = io.BytesIO(); st.session_state.df.to_excel(out, index=False)
            st.download_button("📥 Download Excel", out.getvalue(), "Enriched_Accounts.xlsx")
        with c2:
            if st.button("💬 Send to Slack DM"):
                st.session_state.page = "slack"
                st.rerun()

elif st.session_state.page == "slack":
    st.title("💬 Slack Delivery")
    st.info("📌 **Bot Token Pre-configured:** Only your Member ID is required.")

    uid = st.text_input("Your Member ID (starts with U)").strip()

    if st.button("❓ What is my Member ID?"):
        r_id = requests.get("https://slack.com/api/auth.test", headers={"Authorization": f"Bearer {SLACK_TOKEN}"}).json()
        if r_id.get('ok'):
            st.write(f"Connected to Bot: **{r_id.get('user')}**")
            st.write("To find YOUR ID: Click your profile picture in Slack -> Profile -> More -> Copy Member ID.")
        else: st.error("Bot Connection Error. Check Hardcoded Token.")

    if st.button("🚀 Send Report to my DM"):
        if not uid: st.error("Member ID required.")
        else:
            headers = {"Authorization": f"Bearer {SLACK_TOKEN}"}
            res_dm = requests.post("https://slack.com/api/conversations.open", headers=headers, json={"users": uid}).json()
            if res_dm.get("ok"):
                target_id = res_dm['channel']['id']
                out = io.BytesIO(); st.session_state.df.to_excel(out, index=False)
                f_bytes = out.getvalue()
                message_text = "Based on the account details shared below, I have attached an Excel sheet with the enriched information."
                r1 = requests.post("https://slack.com/api/files.getUploadURLExternal", headers=headers, data={"filename": "Enriched_Report.xlsx", "length": len(f_bytes)}).json()
                if r1.get("ok"):
                    requests.post(r1['upload_url'], data=f_bytes)
                    payload = {
                        "files": [{"id": r1['file_id'], "title": "Research Report"}],
                        "channel_id": target_id,
                        "initial_comment": message_text
                    }
                    r3 = requests.post("https://slack.com/api/files.completeUploadExternal", headers=headers, json=payload).json()
                    if r3.get("ok"): st.success("✅ Sent! Check your Slack 'Apps' section.")
                    else: st.error(f"❌ Delivery Error: {r3.get('error')}")
                else: st.error(f"❌ Connection Error: {r1.get('error')}")
            else: st.error(f"❌ Could not find user {uid}.")

    if st.button("Back to Results"):
        st.session_state.page = "home"
        st.rerun()
