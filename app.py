import os
import streamlit as st
import warnings

# LangChain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

# =========================
# 🚫 Remover avisos
# =========================
warnings.filterwarnings("ignore")

# =========================
# 🔐 OpenAI API Key (Streamlit Secrets)
# =========================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OPENAI_API_KEY não encontrada em st.secrets.")
    st.stop()

api_key = st.secrets["OPENAI_API_KEY"]

# =========================
# 🎨 Configuração da Página
# =========================
st.set_page_config(
    page_title="Assistente Virtual - Pablo Dantas",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# 🎨 CSS Personalizado
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header { font-size: 2.4rem; font-weight: 800; color: #2563EB; text-align: center; margin-top: 1rem; }
.sub-description { font-size: 1.05rem; font-weight: 500; color: #4B5563; text-align: center; margin-top: -0.5rem; margin-bottom: 2rem; }
.sub-header { font-size: 1.2rem; font-weight: 700; color: #1E3A8A; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #DBEAFE; padding-bottom: 0.4rem; }
.user-message { background-color: #DBEAFE; padding: 0.8rem; border-radius: 10px; margin: 0.3rem 0; text-align: right; color: #1E3A8A; font-size: 0.85rem; font-weight: 500; }
.assistant-message { background-color: #E2E8F0; padding: 0.8rem; border-radius: 10px; margin: 0.3rem 0; text-align: left; color: #111827; font-size: 0.85rem; }
#MainMenu, header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================
# 🧠 Sessão
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# 🔧 Modelos
# =========================
@st.cache_resource
def configurar_modelos():
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500
    )
    return embeddings, llm

embedding_model, llm = configurar_modelos()

# =========================
# 📄 Currículo PDF
# =========================
curriculo_dir = "curriculo_pdf"
caminho_pdf = os.path.join(curriculo_dir, "pablo_resume.pdf")

@st.cache_resource
def carregar_index():
    if not os.path.exists(caminho_pdf):
        st.error("📄 PDF do currículo não encontrado.")
        return None

    loader = PyPDFDirectoryLoader(curriculo_dir)
    documentos = loader.load()

    if not documentos:
        st.error("❌ Nenhum documento carregado.")
        return None

    return FAISS.from_documents(documentos, embedding_model)

index = carregar_index()

# =========================
# 🔤 Prompt
# =========================
template = """
Você é o assistente virtual de Pablo Dantas, profissional de Ciência de Dados e Desenvolvedor Python.

Sempre fale na terceira pessoa. Nunca use "eu", "meu", "minha" ou "nós".

Contexto do currículo:
{context}

Histórico da conversa:
{chat_history}

Pergunta: {input}
Resposta:
"""

def obter_resposta(pergunta):
    if not index:
        return "❌ O sistema de currículo não está disponível."

    prompt = ChatPromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = index.as_retriever(search_kwargs={"k": 2})

    chain = create_retrieval_chain(retriever, document_chain)

    response = chain.invoke({
        "input": pergunta,
        "chat_history": st.session_state.chat_history[-6:]
    })

    st.session_state.chat_history.append((pergunta, response["answer"]))
    st.session_state.chat_history = st.session_state.chat_history[-10:]

    return response["answer"]

# =========================
# 🧩 UI
# =========================
st.markdown('<h1 class="main-header">🤖 Assistente Virtual - Pablo Dantas</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-description">Currículo falante com IA</div>', unsafe_allow_html=True)

# =========================
# 💬 Conversa
# =========================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)

# =========================
# 🗑️ Limpar
# =========================
if st.button("🧹 Limpar Conversa"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

# =========================
# 💭 Input
# =========================
pergunta = st.chat_input("Pergunte algo sobre o Pablo...")
if pergunta:
    st.session_state.messages.append({"role": "user", "content": pergunta})
    resposta = obter_resposta(pergunta)
    st.session_state.messages.append({"role": "assistant", "content": resposta})
    st.rerun()
