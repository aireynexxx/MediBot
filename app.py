import streamlit as st
from modules.retriever import retrieve_similar
from modules.llm import generate_answer

# App config
st.set_page_config(page_title=" MediBot Chat", layout="wide")

# Sidebar
with st.sidebar:
    st.title(" MediBot Assistant")
    st.markdown("Ask health-related questions using your local model.")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []

# Init message state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat handler
if prompt := st.chat_input("Type your question here..."):
    # Show user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant reply using your model
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                context = retrieve_similar(prompt)
                reply = generate_answer(prompt, context, st.session_state.messages)
            except Exception as e:
                reply = f"⚠️ Error: {e}"
        st.markdown(reply)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": reply})
