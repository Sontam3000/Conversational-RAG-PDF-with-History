import streamlit as st
import os 
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory


from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG with Pdf and chat history")
st.write("Upload a PDF and chat with it, retaining the context of the conversation.")

api_key = st.text_input("Enter your Groq API Key", type="password")
if api_key:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)
    session_id = st.text_input("Session ID", value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}
    uploaded_files = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=False)
    if uploaded_files:
        temp_path = f"temp_{uploaded_files.name}"

        with open(temp_path, "wb") as f:
            f.write(uploaded_files.read())

        loader = PyPDFLoader(temp_path)
        documents = loader.load()
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(splits, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})  
        
    contextualize_q_system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question accurately." 
        "Given a chat history latest"
        "user question which might reference context in the history, provide a concise and relevant answer."
    )
            
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Context: {context}\n\nUser Question: {question}")
        ]
    )
    question_rewriter = (
        contextualize_q_prompt
        | llm
        | StrOutputParser()
    )
    from langchain_core.runnables import RunnableLambda

    def history_aware_retrieval(inputs):
        question = inputs["question"]
        # chat_history = inputs["chat_history"]

        # if len(chat_history) == 0:
        #     return retriever.invoke(question)

        # rewritten_question = question_rewriter.invoke({
        #     "question": question,
        #     "chat_history": chat_history
        # })
        # return retriever.invoke(question)
        docs = retriever.invoke(question)
        return "\n\n".join(doc.page_content for doc in docs)

    history_aware_retriever = RunnableLambda(history_aware_retrieval)
    
    rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using the context below.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ]
    )

    rag_chain = (
    {
        "context": history_aware_retriever,
        "question": RunnablePassthrough(),
        "chat_history": RunnableLambda(lambda x: x["chat_history"]),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

    def get_session_history(session:str)-> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]
    
    conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",          
            history_messages_key="chat_history",    
    )
    
    user_input = st.text_input("Your question: ")
    if user_input:
        session_history = get_session_history(session_id)
        
        response = conversational_rag_chain.invoke(
            {
                "question": user_input,
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )
        result = StrOutputParser().parse(response)
        st.write(st.session_state.store) 
        st.success("Response gen sussess:")
        st.write(response)
        st.write(result)
        st.write("Chat History:", session_history.messages )  
else:
    st.warning("Please enter your Groq API Key to proceed.")
    