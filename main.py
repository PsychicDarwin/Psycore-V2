import streamlit as st
import psycore
import argparse
from src.main import (
    PromptStage, Elaborator, RAGElaborator, UserPromptElaboration,
    RAGStage, RAGChatStage, IterativeStage
)

st.markdown("""
    <style>
    /* Style normal links */
    a {
        color: white !important;
        text-decoration: none;
    }
    
    /* Style Streamlit markdown-generated links */
    .markdown-text-container a {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)



def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="psycore CLI")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--proceed", action="store_true", help="Allows program to proceed after preprocessing")
    parser.add_argument("--skip-confirmation", action="store_true", help="Skip confirmation prompts during preprocessing")
    return parser.parse_args()

def initialize_psycore(args):
    """Initializes Psycore instance and related components."""
    psycore_instance = psycore.Psycore(args.config)
    return psycore_instance, RAGElaborator(psycore_instance.elaborator_model)

def process_prompt(base_prompt, psycore_instance, rag_elaborator):
    """Processes user prompt using RAG-based elaboration."""
    prompt_stage = PromptStage(None, psycore_instance.prompt_style)
    elaborated_prompt = rag_elaborator.elaborate(base_prompt)
    chosen_rag_prompt, _ = prompt_stage.decide_between_prompts(base_prompt, elaborated_prompt)
    
    rag_stage = RAGStage(psycore_instance.vdb, 5)
    rag_results = rag_stage.get_rag_prompt_filtered(chosen_rag_prompt, psycore_instance.rag_text_similarity_threshold)
    
    rag_chat_results = psycore_instance.rag_chat.chat(base_prompt, rag_results)
    rag_elaborator.queue_history(rag_chat_results.content)
    
    scores = "\n\nSources: \n"
    for result in rag_results:
        scores = scores + "\n" + "Source: " + str(s3_uri_to_link(result['document_path'])) + "\n\nScore = " + str(result["score"]) + "\n"
    
    return rag_chat_results.content + "\n" + scores

def fetch_response(prompt, psycore_instance, rag_elaborator):
    """Fetches response by processing user prompt."""
    return process_prompt(prompt, psycore_instance, rag_elaborator)

def write_response(role, response, avatar):
    """Writes response to chat interface."""
    with st.chat_message(role, avatar=avatar):
        st.write(response)

def s3_uri_to_link(s3_uri):
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket_name = parts[0]
    object_key = parts[1]
    return f"https://{bucket_name}.s3.amazonaws.com/{object_key}"


def initialize_session(psycore_instance, args):
    """Initializes session with preprocessing and chat setup."""
    if args.preprocess:
        psycore_instance.preprocess(skip_confirmation=args.skip_confirmation)
        if not args.proceed:
            exit(0)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    write_response("assistant", "Hi there! What can I help you with today?", "icons/robot_icon.jpg")

def display_chat_history():
    """Displays stored chat messages."""
    for message in st.session_state.messages:
        role = "user" if message["role"] == "user" else "assistant"
        avatar = "icons/user_icon.jpg" if role == "user" else "icons/robot_icon.jpg"
        write_response(role, message["content"], avatar)

def handle_user_input(psycore_instance, rag_elaborator):
    """Handles user input and responds accordingly."""
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        write_response("user", prompt, "icons/user_icon.jpg")

        response = fetch_response(prompt, psycore_instance, rag_elaborator)
        st.session_state.messages.append({"role": "assistant", "content": response})
        write_response("assistant", response, "icons/robot_icon.jpg")

def main():
    """Main function to run Streamlit chat application."""
    st.title("BDUK Intelligent Assistant")
    
    args = parse_arguments()
    psycore_instance, rag_elaborator = initialize_psycore(args)
    
    initialize_session(psycore_instance, args)
    display_chat_history()
    handle_user_input(psycore_instance, rag_elaborator)

if __name__ == "__main__":
    main()