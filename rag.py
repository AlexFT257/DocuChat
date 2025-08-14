import os
import dotenv
from time import time
import streamlit as st

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader,
)

from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

DB_DOCS_LIMIT = 5
os.environ["USER_AGENT"] = "myagent"

# Stream de respuestas
def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# Index
def load_doc_to_db():
    # cargar segun tipo de archivo
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = [] 
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"No se admite ese tipo de documento: {doc_file.type}")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"Error al cargar el documento {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error al cargar el documento {doc_file.name}: {e}")
                    
                    finally:
                        os.remove(file_path)

                else:
                    st.error(F"Maximo numero de documentos alcanzado ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Documento *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* cargado.", icon="✅")


def initialize_vector_db(docs):
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(google_api_key=st.session_state.gemini_api_key ,model="gemini-embedding-001",task_type="RETRIEVAL_DOCUMENT" ),
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
    )

    # Mantener al menos 20 en memoria
    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    print("Number of collections:", len(collection_names))
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db


def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.popover("no se encontro vector_db")
        st.session_state.vector_db = initialize_vector_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)

# Rag

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "En base a la conversacion anterior, genera una query para buscar informacion relevante a la conversacion, centrate en los mensajes mas recientes"),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """Eres ChatDocu, un asistente que tiene acceso a un RAG con los documentos que carga el usuario.
        Debes responder las preguntas del ususario y tendras acceso al contexto para ayudarte en el proceso, ten en cuenta que este puede no estar relacionado o ser de ayuda.
        Puedes usar tu conocimiento para responder las preguntas del usuario.\n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = ""
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# Funciones extra

def generate_documents_summary(llm):
    """Genera un resumen de todos los documentos cargados"""
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 10})
    docs = retriever.get_relevant_documents("resumen general contenido")
    
    prompt = """
    Genera un resumen conciso de los siguientes documentos, identificando:
    1. Temas principales
    2. Puntos clave de cada documento  
    3. Información relevante para el usuario
    
    Documentos:
    {context}
    """
    
    context = "\n\n".join([f"**{doc.metadata.get('source', 'Documento')}:**\n{doc.page_content}" for doc in docs])
    response = llm.invoke([HumanMessage(content=prompt.format(context=context))])
    return response.content

def compare_documents(llm):
    """Compara los documentos cargados identificando similitudes y diferencias"""
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 15})
    docs = retriever.get_relevant_documents("comparación diferencias similitudes")
    
    prompt = """
    Compara los documentos proporcionados identificando:
    1. Similitudes principales
    2. Diferencias clave
    3. Temas comunes y únicos
    4. Conclusiones de la comparación
    
    Documentos:
    {context}
    """
    
    context = "\n\n".join([f"**{doc.metadata.get('source', 'Documento')}:**\n{doc.page_content}" for doc in docs])
    response = llm.invoke([HumanMessage(content=prompt.format(context=context))])
    return response.content


def classify_document_topics(llm):
    """Clasifica los documentos por temas principales"""
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 10})
    docs = retriever.get_relevant_documents("temas principales categorías")
    
    prompt = """
    Clasifica los documentos por temas principales. Para cada documento identifica:
    1. Tema principal
    2. Subtemas relevantes
    3. Categoría general
    4. Palabras clave
    
    Documentos:
    {context}
    """
    
    context = "\n\n".join([f"**{doc.metadata.get('source', 'Documento')}:**\n{doc.page_content}" for doc in docs])
    response = llm.invoke([HumanMessage(content=prompt.format(context=context))])
    return response.content