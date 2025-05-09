import streamlit as st

def fetch_response(prompt):
    response = "Hello back" # Insert Call to API to Get Response
    return f"{response}"

def write_robot_response(response):
    with st.chat_message("assistant", avatar="icons/robot_icon.jpg"):
        st.write(f"{response}")

def write_user_response(response):
    with st.chat_message("assistant", avatar="icons/user_icon.jpg"):
        st.write(f"{response}")

def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    write_robot_response("Hi there! What can I help you with today?")

def display_chat_history():
    for message in st.session_state.messages:
        if message["role"] == "user":
            write_user_response(message["content"])
        else:
            write_robot_response(message["content"])

def handle_user_input():
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        write_user_response(prompt)

        response = fetch_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        write_robot_response(response)


def main():
    st.title("BDUK Intelligent Assistant")
    initialize_session()
    display_chat_history()
    handle_user_input()

if __name__ == "__main__":
    main()