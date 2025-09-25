import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from deep_translator import GoogleTranslator

load_dotenv()

# ---------------- Language Mapping ----------------
lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Odia": "or"
}

# ---------------- Helper Functions ----------------
def translate_text(text, target_lang="en"):
    """Translate text to target language (for UI/output)."""
    try:
        if not text:
            return ""
        if target_lang == "en":
            return text
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text

def translate_to_english(text):
    """Translate farmer input into English (for LLM)."""
    try:
        if not text:
            return ""
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

# ---------------- LLM Setup ----------------
llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b")
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt = PromptTemplate(
 
    template="""
You are a helpful medical assistant. Provide accurate, reliable, and concise health information. Always remind the user that you are not a doctor and your responses are for educational purposes only. Encourage them to consult a qualified healthcare professional for diagnosis or treatment.maximun 250 worgs only {question}
""",
input_variables=["question"]
)
chain = prompt | model | parser

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="TeleMedCare Chatbot")



# Sidebar language selection
st.sidebar.header("🌐 Language")
selected_lang = st.sidebar.selectbox("Choose Language", list(lang_map.keys()))
st.session_state.language = selected_lang
target = lang_map[st.session_state.language]

def tr(text):
    return translate_text(text, target)

# FarmSevak stays in English always ✅
translations = {
    "title": "TeleMedCare Chatbot",
    "subtitle": tr("Your multi language Health assistant "),
    "ask": tr("Ask your question..."),
    "selected_lang": tr("✅ Selected Language:")
}

# Show UI
st.title(translations["title"])
st.write(translations["subtitle"])
st.write(f"{translations['selected_lang']} {st.session_state.language}")

# ---------------- Chat ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a health assistant.")]

user_input = st.chat_input(translations["ask"])

if user_input:
    # Translate user input → English
    translated_input = translate_to_english(user_input) if st.session_state.language != "English" else user_input

    # Store farmer’s original input
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get LLM response (English, ≤150 words)
    result = chain.invoke({"question": translated_input})

    # Translate back to farmer’s language
    if st.session_state.language != "English":
        try:
            translated_result = translate_text(result, target)
        except Exception:
            translated_result = f"(⚠ Translation failed, showing English)\n\n{result}"
    else:
        translated_result = result

    st.session_state.chat_history.append(AIMessage(content=translated_result))

# ---------------- Display Chat ----------------
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
