import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Streamlit page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("RAG Chatbot with LangChain")
st.write("Ask questions about the content in sample_doc.txt")

# Load and process the document
@st.cache_resource
def load_and_process_document():
    # Load the document
    loader = TextLoader("sample_doc.txt")
    documents = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    
    return vector_store

# Initialize the vector store
vector_store = load_and_process_document()

# Initialize the language model (replace with ChatGrok if desired)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Streamlit input and output
query = st.text_input("Enter your question:", placeholder="What is LangChain used for?")
if query:
    # Get the response and source documents
    result = qa_chain({"query": query})
    answer = result["result"]
    source_docs = result["source_documents"]
    
    # Display the answer
    st.write("**Answer:**")
    st.write(answer)
    
    # Display the retrieved context
    st.write("**Retrieved Context:**")
    for i, doc in enumerate(source_docs, 1):
        st.write(f"**Document {i}:**")
        st.write(doc.page_content)
        st.write("---")

# Instructions for running
st.sidebar.write("### Instructions")
st.sidebar.write("1. Ensure `sample_doc.txt` is in the same folder as this script.")
st.sidebar.write("2. Install dependencies: `pip install langchain langchain-community faiss-cpu streamlit`")
st.sidebar.write("3. Run the app: `streamlit run rag_chatbot.py`")
st.sidebar.write("4. Try questions like 'What is LangChain used for?' or 'What is RAG?'")
st.sidebar.write("5. Edit `sample_doc.txt` to add new content and rerun.")