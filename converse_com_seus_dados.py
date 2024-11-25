# Langchain Core
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder

# Langchain Community
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Langchain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Converse com seus Documentos 📚", 
                   page_icon="🦜🔗")

st.title("🦜🔗 Converse com seus Documentos")

def config_retriever(uploads: str):
    
    """
    Cria um retriever a partir de arquivos PDF.

    Os arquivos PDF devem ser carregados por meio do parâmetro `uploads`.
    O retriever é salvo localmente no diretório `vectorstore` e

    Retorna o retriever.
    """
    # Cria uma lista de documento do upload
    with st.spinner(text="This should take 5-6 minutes."):
        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        for file in uploads:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(temp_filepath)
            docs.extend(loader.load())

        # 2. Divide em Chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Tamanho do chunk em caracteres
            chunk_overlap=200)  # Overlap entre os chunks em caracteres

        documentes = splitter.split_documents(docs)

        # 3. Cria o embedding
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        # 4. Cria o vectorstore
        vectorstore = FAISS.from_documents(documentes, embeddings)

        # 5. Salva o vectorstore localmente
        vectorstore.save_local("vectorstore/faiss_db")

        # 6. Cria o retriever
        retriever = vectorstore.as_retriever(search_type="mmr",
                                            search_kwargs={"k": 3, 
                                                           "fetch_k": 4})

    # Retorna o retriever
    return retriever

def make_model(model , temperature):
    llm = ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=512,
        timeout=None,
        max_retries=2,
    )
    return llm

def create_rag_chain(llm, retriever):
    """
    Create a RAG (Retrieve and Generate) chain using the given LLM and retriever.

    A RAG chain is a type of chain that combines the capabilities of a retriever
    (which fetches relevant context from a database) with a language model (which
    generates text based on the retrieved context).

    The RAG chain created by this function will use the given LLM to generate
    text based on the context retrieved by the given retriever.

    The context retrieved by the retriever will be passed to the LLM as input,
    and the LLM will generate text based on that input. The generated text will
    then be returned as the output of the RAG chain.

    The RAG chain is useful for tasks such as answering questions based on a
    database of information, or generating text based on a set of input
    documents.

    :param llm: The LLM to use for generating text.
    :param retriever: The retriever to use for fetching context.
    :return: A RAG chain that uses the given LLM and retriever.
    """

    context_q_system_prompt = """
    Given the following chat history and the follow-up question which might
    reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is.
    """

    # Create a template for the contextualization prompt
    context_q_user_prompt = "Question: {input}"
    context_q_prompt = ChatPromptTemplate.from_messages(
        [("system", context_q_system_prompt), 
         MessagesPlaceholder("chat_history"), 
         ("human", context_q_user_prompt)]
    )

    # Chain para contextualização
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=context_q_prompt
    )

    # Create a prompt for Q&A
    qa_prompt_template = """
    You are a legal advisor in a military organization in a barracks in Brazil.
    Your task is to help the military by answering questions, always using the following 
    retrieved context snippets to answer the question...
    If you don't know the answer, just say you don't know. Keep the answer concise.
    Use the fewest words possible to clarify the military's doubt.

    Always respond in Portuguese:
    \n\n
    Question: {input}
    \n
    Context: {context}
    """

    # Create a template for the Q&A prompt
    qa_prompt = PromptTemplate.from_template(qa_prompt_template)

    # Configure LLM and Chain for Q&A
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create RAG Chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain,
    )

    return rag_chain

# Cria painel lateral na interface
uploads = st.file_uploader(
    label="Enviar arquivos", type=["pdf"],
    accept_multiple_files=True
)

if not uploads:
    st.info("Por favor, envie algum arquivo para continuar!")
    st.stop()

def clear_chat_history():
    st.session_state.chat_history = [{"role": "assistant", 
                                  "content": "Como posso ajudar você hoje?"}]

st.sidebar.title("Pergunte para seu documento📚")

temperature = st.sidebar.slider('temperature', min_value=0.1, max_value=1.0, value=0.1, step=0.01)
model = "claude-3-haiku-20240307"
llm = make_model(model, temperature)

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, pergunte para seu documento"),
    ]

if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🤖"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="🤷"):
            st.write(message.content)

user_query = st.chat_input("Digite sua dúvida aqui...", max_chars=500)
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        with st.spinner("Reunindo fontes..."):
            if st.session_state.docs_list != uploads:
                st.session_state.docs_list = uploads
                st.session_state.retriever = config_retriever(uploads)

            rag_chain = create_rag_chain(llm, st.session_state.retriever)
            result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
            resp = result['answer']
        
        st.write(resp)

        # mostrar a fonte
        st.markdown("**🌐 Resposta baseada nas seguintes fontes:**")

        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page', 'Página não especificada')

            ref = f":link: Fonte {idx + 1}: *{file} - p. {page}*"
            with st.popover(ref):
                st.caption(doc.page_content)

        st.session_state.chat_history.append(AIMessage(content=resp))