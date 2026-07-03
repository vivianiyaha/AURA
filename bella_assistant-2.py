"""
Bella — A Streamlit Personal Assistant
========================================
A chat-first personal assistant built with Streamlit + the Anthropic API,
with optional voice input/output and a safe, allow-listed device-control panel.

WHAT THIS APP ACTUALLY DOES:
- Real chat with an LLM (Anthropic Claude by default; OpenAI or Google Gemini
  are optional alternatives), with memory of the conversation plus a simple
  personal knowledge base (notes) that's injected into context.
- Real to-do list / reminders, stored in session state.
- Optional voice input (record -> transcribe -> send as a chat message) and
  voice output (reply -> speech -> played back), both using free/best-effort
  services that only activate if the relevant package is installed.
- A "Smart Home" panel that is SIMULATED — toggles/sliders that update
  in-memory state only. It is not wired to any real devices.
- A "Device Control" panel that can run a small FIXED allow-list of local
  actions (open calculator, open browser, open text editor, lock screen).
  Every action is a hardcoded command, never built from free-form text, and
  every action requires you to click a real button — chat can never trigger
  these on its own. This app does not execute arbitrary OS commands.

DEPLOYING ON STREAMLIT COMMUNITY CLOUD:
- Push this file + requirements.txt to a GitHub repo, then deploy from
  https://share.streamlit.io using the SAME account that owns the repo.
- Add your API key as a Secret (Settings -> Secrets) rather than typing it
  in the sidebar every time, e.g.:
      ANTHROPIC_API_KEY = "sk-ant-..."
- If you don't need voice features, you can delete the mic-recorder /
  SpeechRecognition / gTTS lines from requirements.txt — the app runs fine
  without them and just hides those controls.

LOCAL SETUP:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...      (or paste it in Settings)
    streamlit run bella_assistant.py
"""

import io
import json
import os
import platform
import subprocess
from datetime import datetime, date

import streamlit as st

# ----------------------------------------------------------------------
# OPTIONAL DEPENDENCIES — the app must not crash if any of these are absent.
# ----------------------------------------------------------------------
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

try:
    from google import genai as google_genai
except ImportError:
    google_genai = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o"
GOOGLE_MODEL = "gemini-2.0-flash"

BELLA_SYSTEM_PROMPT = """You are Bella, a warm, efficient personal assistant living inside a \
Streamlit app. Your tone is crisp, friendly, and to the point — you match the user's energy \
rather than defaulting to corporate-speak.

You have access to the user's saved notes and to-do list, which will be included below when \
relevant. Use them to stay consistent with things the user has told you to remember.

Be honest about what you can and can't do. You can chat, help the user think things through, \
draft text, do quick reasoning/calculations, and read/reference the notes and to-dos they've \
saved in this app. You do NOT have the ability to control real smart-home devices, access the \
internet, send real emails, or take actions outside this chat window — if the user asks for one \
of those, say so plainly and suggest the closest thing you can actually help with (e.g. drafting \
the email for them to send, or adding a reminder to the to-do list).

Keep replies scannable: short paragraphs, bullets when listing things, bold sparingly for emphasis."""

# Fixed allow-list for Device Control — no free-form command execution.
# Each entry maps to a concrete, known command per OS. Nothing here is
# ever built from user- or LLM-supplied text.
DEVICE_ACTIONS = {
    "Open Calculator": {
        "Windows": ["calc.exe"],
        "Darwin": ["open", "-a", "Calculator"],
        "Linux": ["gnome-calculator"],
    },
    "Open Text Editor": {
        "Windows": ["notepad.exe"],
        "Darwin": ["open", "-a", "TextEdit"],
        "Linux": ["gedit"],
    },
    "Open Default Browser": {
        "Windows": ["cmd", "/c", "start", ""],
        "Darwin": ["open", "-a", "Safari"],
        "Linux": ["xdg-open", "https://www.google.com"],
    },
    "Lock Screen": {
        "Windows": ["rundll32.exe", "user32.dll,LockWorkStation"],
        "Darwin": ["pmset", "displaysleepnow"],
        "Linux": ["loginctl", "lock-session"],
    },
}


# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
def init_state():
    defaults = {
        "messages": [],
        "notes": [],
        "todos": [],
        "smart_home": {
            "living_room_lights": {"on": False, "brightness": 70},
            "bedroom_lights": {"on": False, "brightness": 50},
            "thermostat_f": 70,
            "front_door_locked": True,
        },
        "anthropic_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "openai_key": os.environ.get("OPENAI_API_KEY", ""),
        "google_key": os.environ.get("GOOGLE_API_KEY", ""),
        "provider": "Anthropic (Claude)",
        "device_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def next_id(items):
    return (max([i["id"] for i in items], default=0)) + 1


# ----------------------------------------------------------------------
# AI BRAIN
# ----------------------------------------------------------------------
def build_context_block():
    parts = []
    if st.session_state.notes:
        notes_txt = "\n".join("- " + n["text"] for n in st.session_state.notes)
        parts.append("SAVED NOTES:\n" + notes_txt)
    if st.session_state.todos:
        open_todos = [t for t in st.session_state.todos if not t["done"]]
        if open_todos:
            todos_txt = "\n".join(
                "- " + t["text"] + (" (due " + t["due"] + ")" if t["due"] else "")
                for t in open_todos
            )
            parts.append("OPEN TO-DOS:\n" + todos_txt)
    return "\n\n".join(parts)


def call_bella(user_message):
    context = build_context_block()
    system = BELLA_SYSTEM_PROMPT
    if context:
        system += "\n\n---\nCURRENT USER DATA\n" + context

    provider = st.session_state.provider

    if provider == "Anthropic (Claude)":
        if not anthropic:
            return "The `anthropic` package isn't installed. Run `pip install anthropic`."
        if not st.session_state.anthropic_key:
            return "No Anthropic API key set. Add one in Settings to chat with me."
        try:
            client = anthropic.Anthropic(api_key=st.session_state.anthropic_key)
            api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            api_messages.append({"role": "user", "content": user_message})
            response = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=1024, system=system, messages=api_messages
            )
            return "".join(b.text for b in response.content if b.type == "text")
        except Exception as e:
            return "Error talking to the Anthropic API: " + str(e)

    elif provider == "OpenAI":
        if not openai:
            return "The `openai` package isn't installed. Run `pip install openai`."
        if not st.session_state.openai_key:
            return "No OpenAI API key set. Add one in Settings to chat with me."
        try:
            client = openai.OpenAI(api_key=st.session_state.openai_key)
            api_messages = [{"role": "system", "content": system}]
            api_messages += [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            api_messages.append({"role": "user", "content": user_message})
            response = client.chat.completions.create(
                model=OPENAI_MODEL, messages=api_messages, max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return "Error talking to the OpenAI API: " + str(e)

    elif provider == "Google (Gemini)":
        if not google_genai:
            return "The `google-genai` package isn't installed. Run `pip install google-genai`."
        if not st.session_state.google_key:
            return "No Google API key set. Add one in Settings to chat with me."
        try:
            client = google_genai.Client(api_key=st.session_state.google_key)
            history_txt = "\n".join(m["role"] + ": " + m["content"] for m in st.session_state.messages)
            full_prompt = system + "\n\n" + history_txt + "\nuser: " + user_message + "\nassistant:"
            response = client.models.generate_content(model=GOOGLE_MODEL, contents=full_prompt)
            return response.text
        except Exception as e:
            return "Error talking to the Google API: " + str(e)

    return "Unknown provider selected."


# ----------------------------------------------------------------------
# VOICE HELPERS
# ----------------------------------------------------------------------
def transcribe_audio(audio_bytes):
    if not sr:
        return ""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        st.warning("Transcription failed: " + str(e))
        return ""


def synthesize_speech(text):
    if not gTTS or not text.strip():
        return None
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.warning("Speech synthesis failed: " + str(e))
        return None


def run_device_action(action_name):
    system_name = platform.system()
    cmd = DEVICE_ACTIONS.get(action_name, {}).get(system_name)
    if not cmd:
        return "'" + action_name + "' isn't supported on " + system_name + "."
    try:
        subprocess.Popen(cmd)
        return "Ran: " + action_name
    except Exception as e:
        return "Couldn't run '" + action_name + "': " + str(e)


# ----------------------------------------------------------------------
# UI: SIDEBAR
# ----------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("Bella")
    st.sidebar.caption("Your personal assistant")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Chat",
            "To-Do & Reminders",
            "Notes",
            "Smart Home (simulated)",
            "Device Control",
            "Settings",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption(datetime.now().strftime("%A, %B %d — %I:%M %p"))
    open_todos = len([t for t in st.session_state.todos if not t["done"]])
    st.sidebar.caption(str(open_todos) + " open to-do(s) · " + str(len(st.session_state.notes)) + " note(s)")

    return page


# ----------------------------------------------------------------------
# UI: CHAT PAGE
# ----------------------------------------------------------------------
def render_chat():
    st.header("Chat with Bella")

    speak_replies = st.toggle("Speak Bella's replies", value=False, disabled=not gTTS)
    if not gTTS:
        st.caption("Install `gTTS` to enable spoken replies.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    voice_text = None
    if mic_recorder and sr:
        st.caption("Tap to record, tap again to stop:")
        audio = mic_recorder(start_prompt="Record", stop_prompt="Stop", key="mic")
        if audio and audio.get("bytes"):
            with st.spinner("Transcribing..."):
                voice_text = transcribe_audio(audio["bytes"])
            if voice_text:
                st.caption("Heard: " + voice_text)
            else:
                st.caption("Didn't catch that -- try again or type instead.")
    elif not mic_recorder:
        st.caption("Install `streamlit-mic-recorder` to enable voice input.")
    elif not sr:
        st.caption("Install `SpeechRecognition` to enable voice input.")

    typed_prompt = st.chat_input("Ask Bella anything, or tell her what to remember...")
    prompt = typed_prompt or voice_text

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = call_bella(prompt)
            st.markdown(reply)
            if speak_replies:
                audio_bytes = synthesize_speech(reply)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------------------------------------------
# UI: TO-DO PAGE
# ----------------------------------------------------------------------
def render_todos():
    st.header("To-Do & Reminders")

    with st.form("add_todo", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        text = col1.text_input("New item", placeholder="e.g. Send the Q3 report")
        due = col2.date_input("Due (optional)", value=None, min_value=date.today())
        if st.form_submit_button("Add") and text.strip():
            st.session_state.todos.append({
                "id": next_id(st.session_state.todos),
                "text": text.strip(),
                "done": False,
                "due": str(due) if due else None,
            })
            st.rerun()

    st.divider()
    open_items = [t for t in st.session_state.todos if not t["done"]]
    done_items = [t for t in st.session_state.todos if t["done"]]

    if not st.session_state.todos:
        st.info("Nothing on your list yet — add something above.")

    for t in open_items:
        c1, c2 = st.columns([5, 1])
        label = t["text"] + ("  ·  due " + t["due"] if t["due"] else "")
        checked = c1.checkbox(label, value=False, key="todo_" + str(t["id"]))
        if c2.button("Delete", key="del_" + str(t["id"])):
            st.session_state.todos = [x for x in st.session_state.todos if x["id"] != t["id"]]
            st.rerun()
        if checked:
            t["done"] = True
            st.rerun()

    if done_items:
        with st.expander("Completed (" + str(len(done_items)) + ")"):
            for t in done_items:
                st.markdown("~~" + t["text"] + "~~")


# ----------------------------------------------------------------------
# UI: NOTES PAGE
# ----------------------------------------------------------------------
def render_notes():
    st.header("Notes (Bella's memory)")
    st.caption("Anything you save here gets fed into Bella's context, so she can reference it in chat.")

    with st.form("add_note", clear_on_submit=True):
        text = st.text_area("New note", placeholder="e.g. My flight to Denver is on the 14th, gate info TBD")
        if st.form_submit_button("Save note") and text.strip():
            st.session_state.notes.append({
                "id": next_id(st.session_state.notes),
                "text": text.strip(),
                "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            st.rerun()

    st.divider()
    if not st.session_state.notes:
        st.info("No notes saved yet.")
    for n in reversed(st.session_state.notes):
        c1, c2 = st.columns([5, 1])
        c1.markdown("**" + n["created"] + "** — " + n["text"])
        if c2.button("Delete", key="delnote_" + str(n["id"])):
            st.session_state.notes = [x for x in st.session_state.notes if x["id"] != n["id"]]
            st.rerun()


# ----------------------------------------------------------------------
# UI: SMART HOME PAGE (SIMULATED)
# ----------------------------------------------------------------------
def render_smart_home():
    st.header("Smart Home")
    st.warning(
        "This panel is a simulation. These controls update local app state only — "
        "they aren't connected to any real devices. To control real hardware, this would "
        "need to be wired up to your smart-home platform's API (Home Assistant, SmartThings, "
        "Hue, etc.) using your own credentials.",
        icon="⚠️",
    )

    sh = st.session_state.smart_home

    st.subheader("Living Room")
    c1, c2 = st.columns(2)
    sh["living_room_lights"]["on"] = c1.toggle("Lights on", value=sh["living_room_lights"]["on"], key="lr_on")
    sh["living_room_lights"]["brightness"] = c2.slider(
        "Brightness", 0, 100, sh["living_room_lights"]["brightness"], key="lr_bright",
        disabled=not sh["living_room_lights"]["on"]
    )

    st.subheader("Bedroom")
    c1, c2 = st.columns(2)
    sh["bedroom_lights"]["on"] = c1.toggle("Lights on", value=sh["bedroom_lights"]["on"], key="br_on")
    sh["bedroom_lights"]["brightness"] = c2.slider(
        "Brightness", 0, 100, sh["bedroom_lights"]["brightness"], key="br_bright",
        disabled=not sh["bedroom_lights"]["on"]
    )

    st.subheader("Climate & Security")
    c1, c2 = st.columns(2)
    sh["thermostat_f"] = c1.slider("Thermostat (°F)", 60, 85, sh["thermostat_f"])
    locked = c2.toggle("Front door locked", value=sh["front_door_locked"])
    sh["front_door_locked"] = locked
    if not locked:
        st.error("Front door is unlocked (simulated)")


# ----------------------------------------------------------------------
# UI: DEVICE CONTROL PAGE
# ----------------------------------------------------------------------
def render_device_control():
    st.header("Device Control")
    st.info(
        "These are the only actions this app can trigger on your machine. Each one runs "
        "a fixed, known command -- nothing here is built from free-form text, and Bella's chat "
        "replies can never trigger these automatically. You always click the button yourself. "
        "Note: this only works when running locally, since a cloud-hosted app has no access to "
        "your device.",
        icon="🔒",
    )
    st.caption("Detected OS: " + platform.system())

    cols = st.columns(2)
    for i, action_name in enumerate(DEVICE_ACTIONS):
        with cols[i % 2]:
            if st.button(action_name, use_container_width=True, key="dev_" + action_name):
                result = run_device_action(action_name)
                st.session_state.device_log.insert(
                    0, datetime.now().strftime("%H:%M:%S") + " -- " + result
                )
                st.rerun()

    st.divider()
    st.subheader("Activity log")
    if not st.session_state.device_log:
        st.caption("No actions taken yet.")
    for entry in st.session_state.device_log[:10]:
        st.text(entry)


# ----------------------------------------------------------------------
# UI: SETTINGS PAGE
# ----------------------------------------------------------------------
def render_settings():
    st.header("Settings")

    st.subheader("AI brain")
    providers = ["Anthropic (Claude)", "OpenAI", "Google (Gemini)"]
    st.session_state.provider = st.selectbox(
        "Provider", providers, index=providers.index(st.session_state.provider)
    )

    if st.session_state.provider == "Anthropic (Claude)":
        st.session_state.anthropic_key = st.text_input(
            "Anthropic API key", value=st.session_state.anthropic_key, type="password",
            help="Or set ANTHROPIC_API_KEY as an environment variable / Streamlit secret."
        )
    elif st.session_state.provider == "OpenAI":
        st.session_state.openai_key = st.text_input(
            "OpenAI API key", value=st.session_state.openai_key, type="password",
            help="Or set OPENAI_API_KEY as an environment variable / Streamlit secret."
        )
    else:
        st.session_state.google_key = st.text_input(
            "Google API key", value=st.session_state.google_key, type="password",
            help="Or set GOOGLE_API_KEY as an environment variable / Streamlit secret."
        )
    st.caption("Keys are kept only in this browser session's memory -- they aren't written to disk.")

    st.divider()
    st.subheader("Export your data")
    export = {
        "notes": st.session_state.notes,
        "todos": st.session_state.todos,
        "messages": st.session_state.messages,
    }
    st.download_button(
        "Download notes, to-dos & chat as JSON",
        data=json.dumps(export, indent=2),
        file_name="bella_export.json",
        mime="application/json",
    )

    st.divider()
    if st.button("Reset all app data", type="secondary"):
        for k in ["messages", "notes", "todos"]:
            st.session_state[k] = []
        st.rerun()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Bella", page_icon="✨", layout="centered")
    init_state()
    page = render_sidebar()

    if page == "Chat":
        render_chat()
    elif page == "To-Do & Reminders":
        render_todos()
    elif page == "Notes":
        render_notes()
    elif page == "Smart Home (simulated)":
        render_smart_home()
    elif page == "Device Control":
        render_device_control()
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()
