import streamlit as st
from dotenv import load_dotenv
from util import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_csv,
    extract_text_from_excel,
    generate_ai_response,
    get_conversation_history
)

load_dotenv()
st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖", layout="centered")

st.sidebar.title("Options")
upload_file = st.sidebar.file_uploader("Upload a file (format should be pdf,txt,excel,csv)")
clear_button = st.sidebar.button("Clear Chat")

if clear_button:
    st.session_state.chat_history = []
    st.session_state.pop('uploaded_file_content', None)
    print("Chat history and uploaded file content cleared.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if upload_file is not None:
    print(f"File uploaded: {upload_file.name}")
    file_content = ""
    if upload_file.type == "application/pdf":
        file_content = extract_text_from_pdf(upload_file)
    elif upload_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        file_content = extract_text_from_docx(upload_file)
    elif upload_file.type == "text/csv":
        file_content = extract_text_from_csv(upload_file)
    elif upload_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        file_content = extract_text_from_excel(upload_file)
    else:
        raw_data = upload_file.read()
        encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                file_content = raw_data.decode(encoding)
                st.sidebar.text(f"File decoded successfully with {encoding}")
                print(f"File decoded with encoding: {encoding}")
                break
            except (UnicodeDecodeError, TypeError):
                print(f"Failed to decode file with encoding: {encoding}")
                continue
        else:
            st.sidebar.error("An error occurred while decoding the file. Please try another file.")
            print("Failed to decode file with all provided encodings.")
    st.session_state.uploaded_file_content = file_content

st.title("Personal AI Assistant 🤖")

for message in st.session_state.chat_history:
    message_type, text = message
    with st.chat_message(message_type):
        st.markdown(text)

if prompt := st.chat_input("Dive deep into your files. Upload to explore."):
    print(prompt)
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    conversation_history = get_conversation_history(st.session_state.chat_history)
    file_content = st.session_state.get('uploaded_file_content', '')  # Retrieve the file content
    input_with_memory = f"{conversation_history}\nFile Content:\n{file_content}\nUser: {prompt}\nAssistant:"
    response = generate_ai_response(input_with_memory)

    if response:
        st.session_state.chat_history.append(("assistant", response))
        with st.chat_message("assistant"):
            st.markdown(response)
        print("Response generated by AI.")
    else:
        error_message = "Sorry, I didn't get that. Can you try again?"
        st.session_state.chat_history.append(("assistant", error_message))
        with st.chat_message("assistant"):
            st.markdown(error_message)
        print("AI failed to generate a response.")
