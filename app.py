import streamlit as st
from main import transcribe_audio, generate_response, generate_audio
from prompt_managements import pm
import io
import numpy as np
import soundfile as sf

# Constants
DEFAULT_VOICE = "af_heart"
MODEL_CONTEXT = "openai/gpt-oss-20b"
MODEL_CHAT = "openai/gpt-oss-120b"

VOICES = {
    "American Woman 1": "af_heart",
    "American Woman 2": "af_bella",
    "American Man": "am_fenrir",
    "British Woman": "bf_emma",
    "British Man": "bm_fable"
}

# Pre-built ESL scenario library
SCENARIOS = {
    "-- Custom (type your own) --": "",
    "✈️ Airport check-in": "At a busy airport check-in counter, I am a traveller with overweight luggage negotiating with the airline staff.",
    "🛒 Supermarket shopping": "At a supermarket, I am a new arrival trying to find specific ingredients and asking staff for help.",
    "🍽️ Restaurant order": "At a restaurant, I am a customer trying to order food, ask about allergens, and understand the specials.",
    "🏨 Hotel check-in": "At a hotel front desk, I am a guest checking in and asking about hotel facilities and local attractions.",
    "💼 Job interview": "In a formal office, I am a job applicant being interviewed for an entry-level position.",
    "🏥 Doctor's appointment": "At a medical clinic, I am a patient describing symptoms to a doctor and asking about treatment options.",
    "🚌 Asking for directions": "On a busy city street, I am a tourist who is lost and needs to ask a local for directions to a landmark.",
    "📞 Phone call complaint": "On a customer service phone call, I am a customer politely complaining about a faulty product and seeking a refund.",
    "🏦 Bank visit": "At a bank branch, I am a new customer trying to open an account and understand the different options.",
    "🎓 University enrolment": "At a university admin office, I am an international student trying to enrol and asking questions about the process.",
}


def init_session_state() -> None:
    """Initialize session state variables if they don't exist"""
    if "context" not in st.session_state:
        st.session_state.context = ""
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "voice" not in st.session_state:
        st.session_state.voice = DEFAULT_VOICE
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = "Intermediate"


def audio_to_wav_bytes(audio_data, sample_rate=24000):
    """Convert audio data (numpy array or bytes) to WAV bytes for Streamlit playback"""
    try:
        # If audio is already bytes, try to use directly
        if isinstance(audio_data, bytes):
            # Check if it's already a valid WAV
            if audio_data[:4] == b'RIFF':
                return audio_data
            # If it's raw audio bytes, convert to numpy array
            try:
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                # Create WAV in memory
                buffer = io.BytesIO()
                sf.write(buffer, audio_array, sample_rate, format='WAV')
                buffer.seek(0)
                return buffer.read()
            except:
                return audio_data
        
        # If audio is a numpy array
        if isinstance(audio_data, np.ndarray):
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format='WAV')
            buffer.seek(0)
            return buffer.read()
        
        # If audio is a file path
        if isinstance(audio_data, str):
            with open(audio_data, 'rb') as f:
                return f.read()
        
        # If audio is a list
        if isinstance(audio_data, list):
            audio_array = np.array(audio_data, dtype=np.float32)
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='WAV')
            buffer.seek(0)
            return buffer.read()
            
    except Exception as e:
        st.error(f"Audio conversion error: {str(e)}")
        return audio_data if isinstance(audio_data, bytes) else b''
    
    return audio_data if isinstance(audio_data, bytes) else b''


def display_chat_history() -> None:
    """Display the chat history with audio playback"""
    for msg in st.session_state.chat:
        with st.container(border=True):
            role_label = "**🧑 Me**" if msg["role"] == "me" else "**🤖 Assistant**"
            st.write(role_label)
            
            # Convert and play audio
            try:
                audio_bytes = audio_to_wav_bytes(msg["audio"])
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                else:
                    st.warning("No audio data available")
            except Exception as e:
                st.error(f"Could not play audio: {str(e)}")
            
            with st.expander("Show transcript", expanded=False):
                st.write(f"{msg['content']}")


def format_chat_history() -> str:
    """Format the chat history for prompt context"""
    return "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in st.session_state.chat
    )


def generate_context(prompt: str, api_key: str) -> None:
    """Generate context based on a provided prompt and API key"""
    if not api_key:
        st.error("Please enter your Groq API key to generate the context.")
        return
    with st.spinner("Generating scenario..."):
        st.session_state.context = generate_response(prompt, MODEL_CONTEXT, api_key)


def export_conversation() -> str:
    """Export the conversation as a plain text transcript"""
    lines = []
    lines.append("DISCUTE - Conversation Export")
    lines.append("=" * 40)
    if st.session_state.context:
        lines.append(f"\nSCENARIO:\n{st.session_state.context}\n")
    lines.append("CONVERSATION:")
    for msg in st.session_state.chat:
        role = "Me" if msg["role"] == "me" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def main():
    """Main application function"""
    st.set_page_config(page_title="Discute", page_icon="💬", layout="centered")

    # App header
    st.write("# 💬 Discute")
    st.caption("Practice your English speaking skills with an AI conversation partner.")

    # API key — use Streamlit secret if available, otherwise show input
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        st.success("✅ Groq API key loaded from secrets.", icon="🔑")
    except (KeyError, FileNotFoundError):
        groq_api_key = st.text_input(
            "Enter your Groq API key [Get one here](https://console.groq.com/home)",
            type="password",
            value=st.session_state.get("groq_api_key", ""),
        )
        if groq_api_key:
            st.session_state.groq_api_key = groq_api_key

    # Initialize session state
    init_session_state()

    st.divider()

    # --- Scenario / Context section ---
    st.subheader("1. Choose a scenario")

    # Difficulty selector
    difficulty = st.select_slider(
        "Difficulty level",
        options=["Beginner", "Intermediate", "Advanced"],
        value=st.session_state.difficulty,
    )
    st.session_state.difficulty = difficulty

    # Scenario library dropdown
    scenario_choice = st.selectbox(
        "Pick a scenario or write your own below",
        options=list(SCENARIOS.keys()),
    )
    prefill = SCENARIOS[scenario_choice]

    col1, col2 = st.columns(2, border=True)
    with col1:
        st.write("**Custom situation**")
        situation = st.text_input(
            "Describe the situation",
            placeholder="e.g. Ordering food at a cafe",
            value=prefill,
        )
        context_prompt = pm.get_prompt(
            "context_prompt",
            variables={"Situation": situation, "Difficulty": difficulty},
        )
        if st.button("Generate from situation", use_container_width=True):
            generate_context(context_prompt, groq_api_key)

    with col2:
        st.write("**Random scenario**")
        st.write("Let the AI surprise you with a random everyday situation.")
        if st.button("🎲 Generate random", use_container_width=True):
            generate_context(
                pm.get_prompt("random_context", variables={"Difficulty": difficulty}),
                groq_api_key,
            )

    if st.session_state.context:
        st.info(f"📍 **Scenario:** {st.session_state.context}")

    st.divider()

    # --- Voice selection ---
    st.subheader("2. Choose a voice")
    voice_choice = st.selectbox(
        "Select AI voice",
        options=list(VOICES.keys()),
        index=0,
    )
    st.session_state.voice = VOICES[voice_choice]

    st.divider()

    # --- Conversation ---
    st.subheader("3. Have the conversation")

    # Toolbar: clear conversation + export
    tool_col1, tool_col2 = st.columns([1, 1])
    with tool_col1:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.chat = []
            st.rerun()
    with tool_col2:
        if st.session_state.chat:
            transcript = export_conversation()
            st.download_button(
                "⬇️ Export transcript",
                data=transcript,
                file_name="discute_conversation.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # Chat history
    display_chat_history()

    # Audio input
    audio_col, btn_col = st.columns([3, 1])
    with audio_col:
        audio_value = st.audio_input("Record a voice message")
    with btn_col:
        st.write("")
        st.write("")
        st.write("")
        if st.button("Send", use_container_width=True):
            if not audio_value:
                st.error("Please record a voice message before sending.")
            elif not groq_api_key:
                st.error("Please enter your Groq API key before sending.")
            elif not st.session_state.context:
                st.error("Please generate a scenario first.")
            else:
                with st.spinner("Transcribing your voice..."):
                    audio_bytes = audio_value.read()
                    text = transcribe_audio(audio_bytes)
                
                # Convert user audio for playback
                user_audio = audio_to_wav_bytes(audio_bytes)
                
                st.session_state.chat.append(
                    {"role": "me", "content": text, "audio": user_audio}
                )

                chat_history = format_chat_history()
                prompt_vars = {
                    "Context": st.session_state.context,
                    "ChatHistory": chat_history,
                    "Difficulty": st.session_state.difficulty,
                }
                chat_prompt = pm.get_prompt("chat_prompt", variables=prompt_vars)

                with st.spinner("AI is thinking..."):
                    ai_response = generate_response(chat_prompt, MODEL_CHAT, groq_api_key)

                with st.spinner("Generating audio response..."):
                    audio = generate_audio(ai_response, st.session_state.voice)
                    # Convert AI audio for playback
                    ai_audio = audio_to_wav_bytes(audio)

                st.session_state.chat.append(
                    {"role": "you", "content": ai_response, "audio": ai_audio}
                )
                st.rerun()

    st.divider()

    # --- AI Coach Review ---
    st.subheader("4. Get feedback")
    if st.button("🎓 Review my English", use_container_width=True):
        if not st.session_state.chat:
            st.error("No conversation to review yet.")
        elif not groq_api_key:
            st.error("Please enter your Groq API key for the review.")
        else:
            conversation = format_chat_history()
            coach_vars = {
                "context": st.session_state.context,
                "conversation": conversation,
                "difficulty": st.session_state.difficulty,
            }
            coach_prompt = pm.get_prompt("english_coach", variables=coach_vars)
            with st.spinner("Your AI coach is reviewing your conversation..."):
                review = generate_response(coach_prompt, MODEL_CHAT, groq_api_key)
            st.write("**Coach Feedback:**")
            st.success(review)


if __name__ == "__main__":
    main()
