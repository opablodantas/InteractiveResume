import os
import streamlit as st
import warnings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# =========================
# 🚫 Remover avisos irrelevantes
# =========================
warnings.filterwarnings("ignore")

# =========================
# 🌎 Carregar variáveis de ambiente 
# =========================
api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY"))

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
    .quick-btn { height: 110px; font-size: 14px; padding: 0.6rem; border-radius: 10px; background: linear-gradient(135deg, #ffffff, #e5e7eb); box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05); border: none; color: #1F2937; font-weight: 600; cursor: pointer; transition: all 0.2s ease; text-align: center; }
    .quick-btn:hover { background-color: #DBEAFE; color: #1D4ED8; box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1); }
    .contact-footer { font-size: 0.85rem; color: #4B5563; text-align: center; margin-top: 2rem; padding: 1.5rem; background: #ffffff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); }
    #MainMenu, header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================
# 🧠 Sessão
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# =========================
# 🔧 Configuração dos Modelos
# =========================
@st.cache_resource
def configurar_modelos():
    embeddings = OpenAIEmbeddings(api_key=api_key)
    llm = OpenAI(api_key=api_key, temperature=0.7, model_name="gpt-3.5-turbo-instruct")
    chain = ConversationChain(llm=llm, memory=st.session_state.memory, verbose=False)
    return embeddings, chain

embedding_model, conversation_chain = configurar_modelos()

# =========================
# 📄 Carregar Currículo (Upload + Leitura com PyPDFDirectoryLoader)
# =========================
curriculo_dir = "curriculo_pdf"
os.makedirs(curriculo_dir, exist_ok=True)

uploaded_file = st.file_uploader("📄 Faça upload do currículo do Pablo (PDF)", type="pdf")
if uploaded_file:
    caminho_pdf = os.path.join(curriculo_dir, uploaded_file.name)
    with open(caminho_pdf, "wb") as f:
        f.write(uploaded_file.read())

@st.cache_resource
def carregar_index():
    loader = PyPDFDirectoryLoader(curriculo_dir)
    documentos = loader.load()
    return FAISS.from_documents(documentos, embedding_model)

index = carregar_index() if os.listdir(curriculo_dir) else None

# =========================
# 🔤 Template de Prompt
# =========================
template = """
Você é o assistente virtual de Pablo Dantas, um profissional de Ciência de Dados e Desenvolvedor Python.

Sua missão é ajudar recrutadores e visitantes a entender as habilidades, experiências e projetos do Pablo.

Sempre fale sobre ele na terceira pessoa. 
Nunca use termos como "eu", "meu", "minha", ou "nós". 
Exemplo correto: "Pablo é um profissional com experiência em...".

Seja objetivo, converse de forma clara e responda com base nas informações do currículo.
Use emojis com moderação e nunca invente dados.
"""

def obter_resposta(pergunta):
    try:
        if index:
            docs = index.similarity_search(pergunta, k=2)
            contexto = "\n".join([doc.page_content[:1000] for doc in docs])
            prompt = f"{template}\n\nContexto do currículo:\n{contexto}\n\nPergunta: {pergunta}\nResposta:"
        else:
            prompt = f"{template}\n\nPergunta: {pergunta}\nResposta:"
        resposta = conversation_chain.run(input=prompt)
        return resposta
    except Exception as e:
        return f"❌ Erro: {str(e)}"

# =========================
# 🧩 Título
# =========================
st.markdown('<h1 class="main-header">🤖 Assistente Virtual - Pablo Dantas</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-description">Sou o assistente do Pablo. Me pergunte algo sobre as experiências e habilidades dele!</div>', unsafe_allow_html=True)

# =========================
# 💡 Perguntas Rápidas
# =========================
st.markdown('<div class="sub-header">💡 Perguntas Rápidas</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

perguntas_rapidas = {
    "projetos": "Quais projetos Pablo já realizou?",
    "objetivo": "Qual o objetivo profissional de Pablo?",
    "habilidades": "Quais são as principais habilidades de Pablo?",
    "formacao": "Qual é a formação acadêmica de Pablo?"
}

def adicionar_pergunta(pergunta):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    resposta = obter_resposta(pergunta)
    st.session_state.messages.append({"role": "assistant", "content": resposta})
    st.rerun()

with col1:
    if st.button("📋 Projetos", use_container_width=True):
        adicionar_pergunta(perguntas_rapidas["projetos"])

with col2:
    if st.button("🎯 Objetivo", use_container_width=True):
        adicionar_pergunta(perguntas_rapidas["objetivo"])

with col3:
    if st.button("🛠️ Habilidades", use_container_width=True):
        adicionar_pergunta(perguntas_rapidas["habilidades"])

with col4:
    if st.button("🎓 Formação", use_container_width=True):
        adicionar_pergunta(perguntas_rapidas["formacao"])

# =========================
# 💬 Conversa
# =========================
st.markdown('<div class="sub-header">💬 Conversa</div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">🧑‍💼 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

pergunta = st.chat_input("Digite sua pergunta...")
if pergunta:
    st.session_state.messages.append({"role": "user", "content": pergunta})
    resposta = obter_resposta(pergunta)
    st.session_state.messages.append({"role": "assistant", "content": resposta})
    st.rerun()

# =========================
# 📬 Rodapé
# =========================
st.markdown("""
<div class="contact-footer">
    <h4 style="color: #2563EB; font-weight: 700;">✨ Transformando dados em soluções inteligentes</h4>
    <p>Aberto a oportunidades em Ciência de Dados e Desenvolvimento</p>
    <p>📧 <a href="mailto:pablodantasevangelista@hotmail.com">pablodantasevangelista@hotmail.com</a> &nbsp; | &nbsp; 🔗 <a href="https://www.linkedin.com/in/pablodantasevangelista/" target="_blank">LinkedIn</a></p>
</div>
""", unsafe_allow_html=True)
