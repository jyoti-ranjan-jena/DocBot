from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnableMap
from PyPDF2 import PdfReader
import google.generativeai as genai
import os
import streamlit as st

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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    You are DocBot, an intelligent assistant designed to assist users based on the provided context from a PDF or multiple PDF files. 
    Your capabilities include answering questions, providing detailed explanations, brainstorming ideas, and offering summaries. 
    Follow these guidelines:

    1. Comprehensiveness: Provide detailed answers, ensuring clarity and completeness.
    2. Contextual Accuracy: Base your responses strictly on the information available in the context. If the answer is not available in the context, clearly state: "The answer is not available in the provided context."
    3. Brainstorming: When asked to brainstorm, generate creative and relevant ideas based on the context.
    4. Integrity: Do not provide incorrect information or make assumptions beyond the given context.

    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        client=genai,
        temperature=0.3,
    )

    chain = (
        RunnableMap(
            {
                "context": lambda x: x["input_documents"],
                "question": lambda x: x["question"],
            }
        )
        | prompt
        | model
    )

    return chain


def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload your PDF and ask me anything: summarize, explain, brainstorm, or answer specific questions.",
        }
    ]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    inputs = {
        "input_documents": "\n".join([doc.page_content for doc in docs]),
        "question": user_question,
    }

    response = chain.invoke(inputs)
    return response.content if hasattr(response, "content") else response


def main():
    st.set_page_config(page_title="DocBot - PDF Chat", page_icon="🤖")

    if api_key:
        genai.configure(api_key=api_key)
    else:
        st.error("API Key not found. Please check your environment variables.")
        st.stop()

    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
            type="PDF",
            label_visibility="visible",
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing your PDF... 📄"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDF processing complete! Now you can ask me questions.")

    st.title("DocBot 🤖 - Chat with PDF files")
    st.write(
        "Welcome to DocBot! Upload your PDF files and ask me anything: summarize, explain, brainstorm, or answer specific questions."
    )
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Upload some PDFs and ask me a question! 📄",
            }
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
                placeholder.markdown(response)
        if response:
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
