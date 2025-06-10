#streamlit run app.py --server.headless false

import streamlit as st
import json
from ask_chatbotLangGraph_openrouter import ChatBot

# Init session state
# Safe Session State Initialization
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_selected" not in st.session_state:
    st.session_state.model_selected = False
if "selected_model_id" not in st.session_state:
    st.session_state.selected_model_id = None
if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = None

st.set_page_config(page_title="LLM Chat with Context", layout="wide")
st.title("🤖 Chat with your PDF (RAG)")
if st.session_state.model_selected:
    st.markdown(f"**🧠 Model in Use:** `{st.session_state.selected_model_name}`")

# Load LLM models from config
with open("ask_config_openrouter.json", "r") as f:
    config_data = json.load(f)
model_options = config_data["llm_model"]["models"]
model_names = list(model_options.keys())

# Step 1: Model Selection (One-time)
if not st.session_state.model_selected:
    selected_model_name = st.selectbox("Choose an LLM model", model_names)
    if st.button("✅ Confirm Model and Start Chat"):
        with st.spinner("Loading model and initializing..."):
            st.session_state.selected_model_id = model_options[selected_model_name]
            st.session_state.chatbot = ChatBot(config_path="ask_config_openrouter.json")
            st.session_state.chatbot.model_id = st.session_state.selected_model_id
            st.session_state.model_selected = True
            st.session_state.selected_model_name = selected_model_name
        st.success(f"✅ Model '{selected_model_name}' selected. You may now chat.")
        st.rerun()  # Ensure fresh UI
    else:
        st.warning("Please confirm your model selection to begin.")
        st.stop()

# Step 2: Chat Interface
# Update history rendering loop
for i, entry in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        if i == len(st.session_state.chat_history) - 1 and entry.get("metrics"):
            st.markdown(entry["metrics"], unsafe_allow_html=True)

# Update chat_input() logic
if query := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("🤖 Thinking..."):
        result = st.session_state.chatbot.ask(query, return_score=True)

        if isinstance(result, tuple):
            if len(result) == 4:
                answer, avg_score, k, cosine_sim = result
            else:
                answer, avg_score, k = result
                cosine_sim = None
        else:
            answer = result
            avg_score = k = cosine_sim = None

    # Prepare the score display only for this turn
    if avg_score is not None and k is not None:
        score_display = f"`Average Vector relevance scores: {avg_score:.4f} (k={k})`"
        if cosine_sim is not None:
            score_display += f" &nbsp; `Answer to Chunk Cosine Similarity: {cosine_sim:.4f}`"
    else:
        score_display = ""

    # Append everything for this turn
    st.session_state.chat_history.append({
        "question": query,
        "answer": answer,
        "metrics": score_display
    })

    # Render latest assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
        if score_display:
            st.markdown(score_display, unsafe_allow_html=True)
