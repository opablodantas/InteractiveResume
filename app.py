# ===============================
# 📦 Imports e Configurações Iniciais
# ===============================
import os
import streamlit as st
import warnings

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

warnings.filterwarnings("ignore")

# ===============================
# 🔐 OpenAI API Key (Streamlit Cloud)
# ===============================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OPENAI_API_KEY não encontrada em st.secrets.")
    st.stop()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ===============================
# 🎨 Configuração da Página
# ===============================
st.set_page_config(
    page_title="Assistente Virtual - Pablo Dantas",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# 🎨 CSS
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    font-size: 2.4rem;
    font-weight: 800;
    color: #2563EB;
    text-align: center;
    margin-top: 1rem;
}

.sub-description {
    font-size: 1.05rem;
    font-weight: 500;
    color: #4B5563;
    text-align: center;
    margin-top: -0.5rem;
    margin-bottom: 2rem;
}

.user-message {
    background-color: #DBEAFE;
    padding: 0.8rem;
    border-radius: 10px;
    margin: 0.3rem 0;
    text-align: right;
    color: #1E3A8A;
    font-size: 0.85rem;
    font-weight: 500;
}

.assistant-message {
    background-color: #E2E8F0;
    padding: 0.8rem;
    border-radius: 10px;
    margin: 0.3rem 0;
    text-align: left;
    color: #111827;
    font-size: 0.85rem;
}

#MainMenu, header, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 🧠 Sessão
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# 🧠 Embeddings + Chroma
# ===============================
persist_directory = "db_curriculo"

embedding_model = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY
)

@st.cache_resource
def carregar_ou_criar_index():
    curriculo_dir = "curriculo_pdf"

    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )

    loader = PyPDFDirectoryLoader(curriculo_dir)
    documentos = loader.load()

    if not documentos:
        st.error("❌ Nenhum documento PDF encontrado.")
        return None

    return Chroma.from_documents(
        documentos,
        embedding_model,
        collection_name="curriculo-index",
        persist_directory=persist_directory
    )

index = carregar_ou_criar_index()

# ===============================
# 🔧 Prompt + LLM
# ===============================
template = """
Você é o assistente virtual de Pablo Dantas, profissional de Ciência de Dados e Desenvolvedor Python.

Sempre fale na terceira pessoa.
Nunca use "eu", "meu", "minha" ou "nós".

Baseie suas respostas exclusivamente nas informações do currículo.
Se algo não constar no currículo, diga claramente que não consta.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

llm = OpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0.7
)

chain = load_qa_chain(
    llm,
    chain_type="stuff",
    prompt=prompt
)

# ===============================
# 🔍 Função RAG
# ===============================
def obter_resposta(pergunta):
    if not index:
        return "❌ O sistema de currículo não está disponível."

    docs = index.similarity_search(pergunta, k=3)

    return chain.run(
        input_documents=docs,
        question=pergunta
    )

# ===============================
# 🧩 UI
# ===============================
st.markdown('<h1 class="main-header">🤖 Assistente Virtual - Pablo Dantas</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-description">Currículo falante com IA</div>', unsafe_allow_html=True)

# ===============================
# 💬 Histórico
# ===============================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)

# ===============================
# 🗑️ Limpar Conversa
# ===============================
if st.button("🧹 Limpar Conversa"):
    st.session_state.messages = []
    st.rerun()

# ===============================
# 💭 Input
# ===============================
pergunta = st.chat_input("Pergunte algo sobre o Pablo...")
if pergunta:
    st.session_state.messages.append({"role": "user", "content": pergunta})
    resposta = obter_resposta(pergunta)
    st.session_state.messages.append({"role": "assistant", "content": resposta})
    st.rerun()
