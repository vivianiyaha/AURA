import streamlit as st
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import os
import webbrowser

st.set_page_config(page_title="Concept: AURA", page_icon="🧠", layout="centered")

st.title("🧠 Concept: AURA")
st.subheader("Adaptive Universal Reasoning Assistant")
st.write("---")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Voice Input Component
st.write("### 🎤 Talk to AURA")
text_input = speech_to_text(
    start_prompt="Click to Speak 🎙️",
    stop_prompt="Stop Recording 🛑",
    language='en',
    use_container_width=True,
    key='aura_speech'
)

# 2. Process Command if Voice is Detected
if text_input:
    st.session_state.messages.append({"role": "user", "text": text_input})
    
    # --- AURA BRAIN (Mock LLM Logic & Device Control) ---
    command = text_input.lower()
    
    if "open youtube" in command:
        response_text = "Opening YouTube for you right now."
        webbrowser.open("https://www.youtube.com")
    elif "weather" in command:
        response_text = "I am currently running in a local environment. Please integrate a weather API to pull live data."
    else:
        response_text = f"I heard you say: '{text_input}'. System integrations are ready for full deployment."
    # ----------------------------------------------------

    st.session_state.messages.append({"role": "aura", "text": response_text})

# 3. Display Chat History & Play Voice Output
st.write("### 💬 Conversation Log")
for msg in reversed(st.session_state.messages):
    if msg["role"] == "user":
        st.chat_message("user").write(msg["text"])
    else:
        st.chat_message("assistant", avatar="🧠").write(msg["text"])
        
        # 4. Voice Output Component (TTS)
        tts = gTTS(text=msg["text"], lang='en', slow=False)
        tts.save("response.mp3")
        st.audio("response.mp3", format="audio/mp3", autoplay=True)
