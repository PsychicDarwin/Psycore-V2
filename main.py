import streamlit as st
import argparse, html
import psycore
from src.main import PromptStage, RAGElaborator, RAGStage
title = "PsyCore"
st.set_page_config(page_title=title, page_icon="🤖", layout="wide")
# ---------- CSS Styling ----------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    min-height: 100vh;
}

.header-bar {
    /* width: 100vw; */
    max-width: 900px;
    margin: 0 auto 2rem auto;
    background: rgba(30, 41, 59, 0.95);
    padding: 1.5rem 2rem 1rem 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.12);
    display: flex;
    flex-direction: column;
    align-items: center;
}
.header-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.header-logo img {
    width: 38px;
    height: 38px;
}
.header-title {
    font-size: 2.2rem;
    color: #fff;
    font-weight: 700;
    letter-spacing: 1px;
}
.header-subtitle {
    color: #cbd5e1;
    font-size: 1.1rem;
    margin-top: 0.25rem;
    font-weight: 400;
}

.chatbox {
    background-color: #232946;
    border-radius: 1.1rem;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    transition: transform 0.2s;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}

.user-bubble {
    background: linear-gradient(90deg, #6366f1 60%, #8b5cf6 100%);
    color: white;
    text-align: right;
    margin-left: auto;
    border-bottom-right-radius: 0.2rem;
    border-bottom-left-radius: 1.1rem;
    border-top-right-radius: 1.1rem;
    border-top-left-radius: 1.1rem;
}

.bot-bubble {
    background: #1e293b;
    color: white;
    margin-right: auto;
    border-bottom-left-radius: 0.2rem;
    border-bottom-right-radius: 1.1rem;
    border-top-right-radius: 1.1rem;
    border-top-left-radius: 1.1rem;
}

.gradient-bar {
    height: 4px;
    border-radius: 9999px;
    background: linear-gradient(to right, #6366f1, #8b5cf6, #ec4899);
    margin-bottom: 0.5rem;
}

.score-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: bold;
    margin-left: 0.5rem;
}

.green { background-color: #22c55e; color: white; }
.yellow { background-color: #eab308; color: white; }
.red { background-color: #ef4444; color: white; }

.source-link {
    color: #60a5fa;
    text-decoration: none;
    font-weight: 500;
}

.stChatInputContainer {
    background: #232946 !important;
    border-radius: 1rem !important;
    padding: 0.7rem 1.2rem !important;
    margin-top: 2rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
}

.stTextInput>div>input {
    background: #1e293b !important;
    color: #fff !important;
    border-radius: 0.7rem !important;
    border: 1.5px solid #6366f1 !important;
    padding: 0.7rem 1rem !important;
    font-size: 1.1rem !important;
}

.stTextInput>div>input:focus {
    border: 2px solid #8b5cf6 !important;
    outline: none !important;
}

.footer-tip {
    color: #cbd5e1;
    text-align: center;
    font-size: 1rem;
    margin-top: 2.5rem;
    margin-bottom: 0.5rem;
    opacity: 0.85;
}

.chat-center-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    /* justify-content: center; */
    /* min-height: 60vh; */
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 900px;
    margin: 0 auto;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header Bar ----------
st.markdown(
    """
    <div class="header-bar">
        <div class="header-logo">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712103.png">
            <span class="header-title">""" + title + """</span>
        </div>
        <div class="header-subtitle">
            Your LLM-powered assistant for BDUK
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Footer Tip ----------
st.markdown(
    """
    <div class="footer-tip">
        💡 Tip: Ask me anything about BDUK or related topics!
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Utility ----------
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--proceed", action="store_true")
    parser.add_argument("--skip-confirmation", action="store_true")
    return parser.parse_args()

def s3_uri_to_link(s3_uri):
    parts = s3_uri.replace("s3://", "").split("/", 1)
    return f"https://{parts[0]}.s3.amazonaws.com/{parts[1]}"

def format_sources(rag_results):
    html_blocks = []
    for result in rag_results:
        score = float(result["score"])
        color = "green" if score >= 0.75 else "yellow" if score >= 0.5 else "red"
        link = s3_uri_to_link(result['document_path'])
        doc_name = result['document_path'].split('/')[-1]
        block = f'''
        <div class='flex justify-between items-center'>
            <a class='source-link' href="{link}" target="_blank">{doc_name}</a>
            <span class="score-pill {color}">{score:.3f}</span>
        </div>
        '''
        html_blocks.append(block)
    return "<br>".join(html_blocks)

def write_response(role, content, sources_block=None):
    alignment = "user-bubble" if role == "user" else "bot-bubble"
    bar = '<div class="gradient-bar"></div>' if role == "assistant" else ""
    sources_html = f"<div class='text-sm mt-2'>{sources_block}</div>" if sources_block else ""

    if role == "user":
        # For user messages, display as plain text to avoid HTML rendering
        st.markdown(f"""
        <div class='chatbox {alignment}'>
            <div style="white-space: pre-wrap;">{html.escape(content)}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # For assistant messages, render HTML normally
        html_block = f"""
        <div class='chatbox {alignment}'>
            {bar}
            <div>{content}</div>
            {sources_html}
        </div>
        """
        st.markdown(html_block, unsafe_allow_html=True)


# ---------- Core Chat Logic ----------
def fetch_response(prompt, psycore_instance, rag_elaborator):
    prompt_stage = PromptStage(None, psycore_instance.prompt_style)
    elaborated_prompt = rag_elaborator.elaborate(prompt)
    chosen_prompt, _ = prompt_stage.decide_between_prompts(prompt, elaborated_prompt)

    rag_stage = RAGStage(psycore_instance.vdb, 5)
    rag_results = rag_stage.get_rag_prompt_filtered(chosen_prompt, psycore_instance.rag_text_similarity_threshold)
    chat_result = psycore_instance.rag_chat.chat(prompt, rag_results)
    rag_elaborator.queue_history(chat_result.content)

    return chat_result.content, format_sources(rag_results)

# ---------- Session State ----------
def initialize_session(psycore_instance, args):
    if args.preprocess:
        psycore_instance.preprocess(skip_confirmation=args.skip_confirmation)
        if not args.proceed:
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        write_response("assistant", "Hi there! Ask me anything to get started.")

def display_chat_history():
    for msg in st.session_state.messages:
        write_response(msg["role"], msg["content"], msg.get("sources", None))

def handle_user_input(psycore_instance, rag_elaborator):
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        write_response("user", prompt)

        response, sources = fetch_response(prompt, psycore_instance, rag_elaborator)
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
        write_response("assistant", response, sources)

# ---------- Main ----------
def main():
    args = parse_arguments()
    psycore_instance = psycore.Psycore(args.config)
    rag_elaborator = RAGElaborator(psycore_instance.elaborator_model)

    initialize_session(psycore_instance, args)
    # Wrap chat history and input in the centering container
    st.markdown(
        '<div class="chat-center-container">',
        unsafe_allow_html=True
    )
    display_chat_history()
    handle_user_input(psycore_instance, rag_elaborator)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
