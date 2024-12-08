import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
You are DocBot, an intelligent assistant designed to answer questions based on the provided context. 
Your responses should be accurate, detailed, and relevant. 
Follow these guidelines:

1. Comprehensiveness: Provide detailed answers, ensuring clarity and completeness.
2. Contextual Accuracy: Base your responses strictly on the information available in the context. If the answer is not available in the context, clearly state: "The answer is not available in the provided context."
3. Integrity: Do not provide incorrect information or make assumptions beyond the given context.

Context:\n {context}?\n
Question:\n {question}\n

Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        client=genai,
        temperature=0.3,
    )
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}
    ]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )  # type: ignore

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )

    print(response)
    return response

def main():
    st.set_page_config(page_title="DocBot - PDF Chat", page_icon="🤖")

    if api_key:
        genai.configure(api_key=api_key)
    else:
        st.error("API Key not found. Please check your environment variables.")
        st.stop()

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    st.title("DocBot 🤖 - Chat with PDF files")
    st.write("Welcome to DocBot! Upload your PDFs and start chatting with the content.")
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question! 📄"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ""
                for item in response["output_text"]:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
