import streamlit as st
import json
from utils.rag_engine import query_llm, retrieve_context

def display_investigator():
    st.markdown("## 🤖 Copilot Investigator")
    st.caption("Offline, privacy-preserving LLM Agent cross-referencing global AML topologies.")
    
    t1, t2, t3 = st.tabs(["🧐 Contextual Breakdown", "💬 Copilot Chat", "📝 Compliance SAR Gen"])
    
    with t1:
        st.markdown("<div class='stCard'><h4>Automated Explainer</h4><p style='color:#666'>Extract causal reasoning for recently scored nodes.</p></div>", unsafe_allow_html=True)
        if "last_tx_scored" in st.session_state:
            ctx = st.session_state["last_tx_scored"]
            st.info(f"Loaded: Transaction profile mapping to **{ctx['tier']}** risk tier.")
            
            if st.button("🚀 Initialize RAG Trace", type="primary"):
                query_str = f"Bitcoin transaction score {ctx['tier']} XGBoost {ctx['results']['xgb']:.4f}"
                context = retrieve_context(query_str)
                
                prompt = f"""
You are an expert FinCEN Bitcoin fraud investigator.

Transaction Details:
- Global Confidence: {ctx['results']['ensemble']:.4f} ({ctx['tier']})
- Sub-models: XGB={ctx['results']['xgb']:.4f}, LGBM={ctx['results']['lgbm']:.4f}, CB={ctx['results']['cb']:.4f}
- Core Anomalies Simulated:
{json.dumps(ctx['top_features'], indent=2)}

Knowledge Context:
{context}

Draft a clear intelligence report explaining:
1. Why the target triggered this tier
2. Real-world mapping of these variables
3. Whether it aligns with Darknet, Ponzi, or mixing topologies
"""
                with st.spinner("Compiling tactical rationale..."):
                    st.write_stream(query_llm(prompt))
        else:
            st.warning("Memory buffer is empty. Evaluate a trace inside 'Score Transaction' first.")
            
    with t2:
        st.markdown("### Analyst Chat Interface")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Welcome. I am the BitcoinGuard Copilot. How can I assist your investigation today?"}
            ]
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧐"):
                st.markdown(msg["content"])
                
        if q := st.chat_input("Query FinCEN AML typologies or structural logic..."):
            st.session_state.messages.append({"role": "user", "content": q})
            
            with st.chat_message("user", avatar="🧐"):
                st.markdown(q)
                
            with st.chat_message("assistant", avatar="🤖"):
                context = retrieve_context(q)
                
                prompt = f"""
You are an enterprise Bitcoin fraud analyst. Combine the Context with your insight to answer accurately.

Context:
{context}

Question:
{q}
"""             
                response = st.write_stream(query_llm(prompt))
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                with st.expander("🔗 Reference Materials", expanded=False):
                    st.markdown(context)
                    
    with t3:
        st.markdown("<div class='stCard' style='border-left: 5px solid #2980b9;'><h4>Automated Suspicious Activity Report (SAR)</h4><p>Produces a standardized markdown output ready for compliance transmission.</p></div>", unsafe_allow_html=True)
        
        if "last_tx_scored" in st.session_state:
            ctx = st.session_state["last_tx_scored"]
            if st.button("Generate SAR Bundle", type="primary"):
                prompt = f"""
Draft a Suspicious Activity Report (SAR).

Include:
- Summary Highlights
- Transaction Scores ({ctx['tier']} at {ctx['results']['ensemble']:.4f})
- Machine Learning Evidence
- Recommended FATF Actions

Use professional compliance language and bold headers.
"""
                st.write_stream(query_llm(prompt))
        else:
            st.warning("You must evaluate a transaction first to generate an associated SAR.")

display_investigator()
