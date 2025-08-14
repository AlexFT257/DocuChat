import streamlit as st
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
import uuid
import os
import dotenv
import asyncio

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


dotenv.load_dotenv()

from rag import (
    load_doc_to_db,
    stream_llm_response,
    stream_llm_rag_response,
    generate_documents_summary,
    compare_documents,
    classify_document_topics
)

st.set_page_config(
    page_title="DocuChat",
    page_icon="📄",
)

st.write("# Bienvenido a DocuChat")

st.markdown(
    """
    DocuChat es una aplicacion que permite cargar documentos para analizar, comparar y conversar
    con tus ellos. Debido a limitaciones del tier de Gemini si se **suben multiples archivos de manera consecutiva se pueden producir errores**. Esta aplicacion es parte de una prueba tecnica para el puesto de Desarrollador/a Joven de IA de Catchai.
    """
)


st.divider()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

with st.sidebar:
    gemini_api_key = st.text_input("Gemini API Key", key="file_qa_api_key", type="password")
    "[Obten tu API Key](https://aistudio.google.com/app/apikey)"

   
if not gemini_api_key:
    st.warning("Ingresa tu Api Key de Gemini")
else:
    st.session_state.gemini_api_key= gemini_api_key
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Sube un documento",
            accept_multiple_files=True,
            type=("txt", "md", "pdf"),
            on_change=load_doc_to_db(),
            key="rag_docs"
        )

        # Validación en load_doc_to_db:
        if len(uploaded_file) > 5:
            st.error("Máximo 5 documentos PDF permitidos")
            uploaded_file=uploaded_file[:4]

        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "Use RAG", 
                value=is_vector_db_loaded, 
                key="use_rag", 
                disabled=not is_vector_db_loaded,
            )
        
        with st.expander(f"Documentos en la BD ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])
    
    

    llm_stream = ChatGoogleGenerativeAI(
        api_key=gemini_api_key,
        model="gemini-2.5-flash",
        temperature=0.3,
    )

    with st.sidebar:
         if is_vector_db_loaded and len(st.session_state.rag_sources) > 0:
        
            # Botón para generar resumen
            if st.button("Generar Resumen de Documentos"):
                with st.spinner("Generando resumen..."):
                    summary = generate_documents_summary(llm_stream)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"**Resumen de documentos cargados:**\n\n{summary}"
                    })
                    st.rerun()
            
            # Botón para comparar documentos
            if len(st.session_state.rag_sources) > 1:
                if st.button("Comparar Documentos"):
                    with st.spinner("Comparando documentos..."):
                        comparison = compare_documents(llm_stream)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"**Comparación entre documentos:**\n\n{comparison}"
                        })
                        st.rerun()
            
            # Clasificación por temas
            if st.button("Clasificar por Temas"):
                with st.spinner("Clasificando temas..."):
                    topics = classify_document_topics(llm_stream)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"**Clasificación temática:**\n\n{topics}"
                    })
                    st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hola, soy DocuChat ¿Que consulta tienes el dia de hoy? Recuerda que puesdes subir documentos en la barra lateral."}
    ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
            if not st.session_state.use_rag:
                st.write_stream(stream_llm_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))