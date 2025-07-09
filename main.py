import streamlit as st
import argparse
import html
import psycore
import time
from src.main import RAGElaborator
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

/* Expandable sections */
.expandable-header {
    cursor: pointer;
    padding: 0.5rem;
    background: rgba(99, 102, 241, 0.1);
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
    transition: background-color 0.2s;
}

.expandable-header:hover {
    background: rgba(99, 102, 241, 0.2);
}

.expandable-content {
    display: none;
    padding: 0.75rem;
    background: rgba(30, 41, 59, 0.3);
    border-radius: 0.5rem;
    margin-top: 0.5rem;
    border-left: 3px solid #6366f1;
}

.expandable-content.show {
    display: block;
}

.expand-icon {
    float: right;
    transition: transform 0.2s;
}

.expand-icon.expanded {
    transform: rotate(90deg);
}

/* Score tooltips */
.score-tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
    border-bottom: 1px dotted #60a5fa;
}

.score-tooltip .tooltip-text {
    visibility: hidden;
    width: 200px;
    background-color: #1e293b;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 0.8rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}

.score-tooltip:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}

.score-detail {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.25rem 0;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2);
}

.score-detail:last-child {
    border-bottom: none;
}

.score-label {
    font-weight: 500;
    color: #cbd5e1;
}

.score-value {
    font-weight: bold;
    color: #60a5fa;
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

.source-container {
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
    margin-top: 1.5rem;
    margin-bottom: 2rem;
}

.score-badge {
    display: inline-block;
    background: linear-gradient(90deg, #22c55e 60%, #16a34a 100%);
    color: #fff;
    border-radius: 9999px;
    padding: 0.18rem 0.85rem;
    font-size: 0.95em;
    font-weight: 600;
    margin-left: 0.5rem;
    margin-right: 0.5rem;
    letter-spacing: 0.5px;
    box-shadow: 0 1px 4px rgba(34,197,94,0.10);
}

.expander-spacer { height: 0.7rem; }
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

EVAL_LABELS = {
    "BERT F1": {
        "label": "BERT F1",
        "desc": "BERTScore F1: Measures how semantically similar the response is to the source text"
    },
    "Graph F1": {
        "label": "Graph F1",
        "desc": "Graph-based F1: Measures how the response relates to the original document, on a relation level"
    },
    "ROUGE-L F1": {
        "label": "ROUGE-L F1",
        "desc": "ROUGE-L F1: Measures similarity between exact text matches"
    }
}

def display_sources(rag_results):
    st.markdown('<div class="source-container">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#cbd5e1; margin-bottom:0.7rem;'>Sources & Evaluation</h4>", unsafe_allow_html=True)
    for idx, result in enumerate(rag_results):
        score = float(result["score"])
        color = "🟢" if score >= 0.75 else "🟡" if score >= 0.5 else "🔴"
        link = s3_uri_to_link(result['document_path'])
        doc_name = result['document_path'].split('/')[-1].replace('_', ' ')
        f1_scores = {}
        for key in EVAL_LABELS:
            if key == "BERT F1" and 'bertscore_evaluation' in result and 'f1' in result['bertscore_evaluation']:
                f1_scores[key] = result['bertscore_evaluation']['f1'][0]
            if key == "Graph F1" and 'graph_evaluation' in result and 'f_beta' in result['graph_evaluation']:
                f1_scores[key] = result['graph_evaluation']['f_beta']
            if key == "ROUGE-L F1" and 'rouge_evaluation' in result and 'rougeL' in result['rouge_evaluation']:
                f1_scores[key] = result['rouge_evaluation']['rougeL'][2]
        avg_f1 = sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0.0

        # Always-visible row with link and badge
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;gap:1rem;margin-bottom:0.2rem;'>
                <a href="{link}" target="_blank" style="text-decoration:none;font-weight:600;color:#60a5fa;font-size:1.05em;">
                    🔗 {doc_name}
                </a>
                <span class="score-badge">{score:.3f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(f"{color} Details & Evaluation", expanded=False):
            st.markdown(f"<div style='margin-bottom:0.7rem;'><b>Average F1:</b> <span style='color:#60a5fa;font-size:1.1em;'>{avg_f1:.3f}</span></div>", unsafe_allow_html=True)
            # Intuitive card/grid for scores
            st.markdown("""
            <div style='display: flex; gap: 1.2rem; flex-wrap: wrap; margin-bottom: 1rem;'>
            """, unsafe_allow_html=True)
            for key, value in f1_scores.items():
                st.markdown(
                    f"""
                    <div style='background:rgba(99,102,241,0.10);border-radius:0.7rem;padding:0.7rem 1.1rem;min-width:170px;box-shadow:0 1px 6px rgba(99,102,241,0.07);margin-bottom:0.5rem;'>
                        <div style='font-weight:600;color:#a5b4fc;font-size:1.05em;margin-bottom:0.2rem;'>{EVAL_LABELS[key]['label']}</div>
                        <div style='font-size:1.25em;font-weight:700;color:#60a5fa;margin-bottom:0.2rem;'>{value:.3f}</div>
                        <div style='font-size:0.97em;color:#cbd5e1;'><span style='font-size:1.1em;'>❓</span> {EVAL_LABELS[key]['desc']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
            # Button to show compared text (no nested expander)
            show_key = f"showtext_{idx}_{result.get('vector_id', result.get('document_path', idx))}"
            if st.button("Show compared text", key=show_key):
                st.info(result.get('text', 'No text available for this source.'))
        if idx < len(rag_results) - 1:
            st.markdown('<div class="expander-spacer"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def fetch_response(prompt, psycore_instance, rag_elaborator):
    start_time = time.time()
    results = psycore_instance.evaluate_prompt(prompt)
    elapsed_time = time.time() - start_time
    response = results["response"]
    rag_results = results["rag_results"]
    response_with_timing = f"{response}\n\n⏱️ **Response generated in {elapsed_time:.2f} seconds**"
    rag_elaborator.queue_history(response)
    return response_with_timing, rag_results

def write_response(role, content, sources_block=None):
    alignment = "user-bubble" if role == "user" else "bot-bubble"
    bar = '<div class="gradient-bar"></div>' if role == "assistant" else ""
    if role == "user":
        st.markdown(f"""
        <div class='chatbox {alignment}'>
            <div style="white-space: pre-wrap;">{html.escape(content)}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        html_block = f"""
        <div class='chatbox {alignment}'>
            {bar}
            <div>{content}</div>
        </div>
        """
        st.markdown(html_block, unsafe_allow_html=True)


# ---------- Core Chat Logic ----------
def display_chat_history():
    for msg in st.session_state.messages:
        write_response(msg["role"], msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            display_sources(msg["sources"])

def handle_user_input(psycore_instance, rag_elaborator):
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        write_response("user", prompt)
        response, rag_results = fetch_response(prompt, psycore_instance, rag_elaborator)
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": rag_results})
        write_response("assistant", response)
        display_sources(rag_results)

# ---------- Main ----------
def main():
    args = parse_arguments()
    psycore_instance = psycore.Psycore(args.config)
    rag_elaborator = RAGElaborator(psycore_instance.elaborator_model)

    if args.preprocess:
        psycore_instance.preprocess(skip_confirmation=args.skip_confirmation)
        if not args.proceed:
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        write_response("assistant", "Hi there! Ask me anything to get started.")

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
