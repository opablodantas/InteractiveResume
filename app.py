# ===============================
# 📦 Imports e Configurações Iniciais
# ===============================
import os
import streamlit as st
import warnings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

.suggestion-box {
    background-color: #F3F4F6;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.suggestion-box:hover {
    background-color: #E5E7EB;
    border-color: #2563EB;
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(37,99,235,0.1);
}

.suggestion-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.suggestion-text {
    font-size: 0.95rem;
    font-weight: 500;
    color: #1F2937;
    line-height: 1.4;
}

.user-message {
    background-color: #DBEAFE;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.3rem 0;
    text-align: right;
    color: #1E3A8A;
    font-size: 0.95rem;
    font-weight: 500;
    border-left: 4px solid #2563EB;
}

.assistant-message {
    background-color: #F3F4F6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.3rem 0;
    text-align: left;
    color: #111827;
    font-size: 0.95rem;
    border-left: 4px solid #6B7280;
}

#MainMenu, header, footer {
    visibility: hidden;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    padding: 0.5rem 2rem;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(37,99,235,0.2);
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
    pdf_path = os.path.join(curriculo_dir, "pablo_resume.pdf")
    
    if not os.path.exists(pdf_path):
        st.error(f"❌ Arquivo não encontrado: {pdf_path}")
        st.error("Certifique-se de que o arquivo 'pablo_resume.pdf' está na pasta 'curriculo_pdf'")
        return None
    
    if os.path.exists(persist_directory):
        try:
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model
            )
        except Exception as e:
            st.warning("⚠️ Recriando índice devido a incompatibilidade...")
            import shutil
            shutil.rmtree(persist_directory)
    
    # Carregar PDF específico
    loader = PyPDFLoader(pdf_path)
    documentos = loader.load()
    
    if not documentos:
        st.error("❌ Não foi possível carregar o PDF.")
        return None
    
    # Dividir texto em chunks para melhor processamento
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    documentos = text_splitter.split_documents(documentos)
    
    # Criar e persistir o índice
    vectorstore = Chroma.from_documents(
        documentos,
        embedding_model,
        collection_name="curriculo-index",
        persist_directory=persist_directory
    )
    
    vectorstore.persist()
    return vectorstore

# ===============================
# 🔧 Prompt + LLM
# ===============================
template = """
Você é o assistente virtual de Pablo Dantas, profissional de Ciência de Dados e Desenvolvedor Python.

Sempre fale na terceira pessoa.
Nunca use "eu", "meu", "minha" ou "nós".

Baseie suas respostas exclusivamente nas informações do currículo fornecido.
Se algo não constar no currículo, diga claramente que não consta.
Seja específico e detalhado quando possível.

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
    temperature=0.5,
    max_tokens=500
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
    index = carregar_ou_criar_index()
    
    if not index:
        return "❌ O sistema de currículo não está disponível."
    
    # Buscar documentos relevantes com maior número de chunks
    docs = index.similarity_search(pergunta, k=5)
    
    if not docs:
        return "Desculpe, não encontrei informações relevantes no currículo para responder sua pergunta."
    
    try:
        resposta = chain.run(
            input_documents=docs,
            question=pergunta
        )
        return resposta
    except Exception as e:
        return f"❌ Erro ao gerar resposta: {str(e)}"

# ===============================
# 🧩 UI
# ===============================
st.markdown('<h1 class="main-header">🤖 Assistente Virtual - Pablo Dantas</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-description">Currículo falante com IA - Pergunte sobre a trajetória profissional de Pablo</div>', unsafe_allow_html=True)

# ===============================
# 💡 Sugestões de Perguntas
# ===============================
st.markdown("### 💡 Sugestões de Perguntas")
col1, col2, col3 = st.columns(3)

sugestoes = [
    {
        "icon": "🏆",
        "texto": "Qual projeto o Pablo sente mais orgulho e o motivo disso?",
        "pergunta": "Qual projeto o Pablo sente mais orgulho e o motivo disso?"
    },
    {
        "icon": "📊",
        "texto": "Qual o impacto de Pablo no trabalho atual?",
        "pergunta": "Qual o impacto de Pablo no trabalho atual?"
    },
    {
        "icon": "🎯",
        "texto": "Qual área Pablo quer trabalhar e por que?",
        "pergunta": "Qual área Pablo quer trabalhar e por que?"
    }
]

for col, sugestao in zip([col1, col2, col3], sugestoes):
    with col:
        # Criar um container clicável
        if st.button(
            f"{sugestao['icon']}\n\n{sugestao['texto']}",
            key=f"sugestao_{sugestao['texto'][:20]}",
            use_container_width=True,
        ):
            # Adicionar a pergunta ao histórico e gerar resposta
            st.session_state.messages.append({"role": "user", "content": sugestao['pergunta']})
            with st.spinner("🔍 Pesquisando no currículo..."):
                resposta = obter_resposta(sugestao['pergunta'])
            st.session_state.messages.append({"role": "assistant", "content": resposta})
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ===============================
# 💬 Histórico
# ===============================
if st.session_state.messages:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">👤 <strong>Você:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🤖 <strong>Assistente:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)

# ===============================
# 🗑️ Limpar Conversa
# ===============================
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🧹 Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ===============================
# 💭 Input
# ===============================
pergunta = st.chat_input("💭 Digite sua pergunta sobre o Pablo...")
if pergunta:
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.spinner("🔍 Pesquisando no currículo..."):
        resposta = obter_resposta(pergunta)
    st.session_state.messages.append({"role": "assistant", "content": resposta})
    st.rerun()