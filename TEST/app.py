import streamlit as st
from openai import OpenAI

# ✅ Initialize OpenAI client properly (make sure this is the right way for your version)
client = OpenAI(api_key="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")

st.title("💬 YAKA v1.1")

# 🗂️ Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

# 📜 Display chat history (excluding system message)
for msg in st.session_state.messages[1:]:
    role = "You" if msg["role"] == "user" else "ChatGPT"
    st.markdown(f"**{role}:** {msg['content']}")

# 💬 User input
user_input = st.text_input("Type your message:")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call OpenAI Chat Completion
    response = client.chat.completions.create(
        model="gpt-4o",  # You can change to "gpt-3.5-turbo" if needed
        messages=st.session_state.messages
    )

    # Extract reply and add to chat
    reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Refresh to show new chat entry
    st.experimental_rerun()
