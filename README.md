

# **Assistente Virtual - Pablo Dantas**

Este projeto consiste em um **assistente virtual interativo**, desenvolvido utilizando **Streamlit**, **LangChain**, **FAISS** e **OpenAI GPT-3.5**. O assistente foi projetado para responder a perguntas sobre o perfil profissional de **Pablo Dantas**, um profissional em **Ciência de Dados** e **Desenvolvimento Python**. O objetivo é fornecer uma experiência interativa para recrutadores ou outros interessados em conhecer as habilidades, experiências e projetos do Pablo de forma rápida e eficiente.

## **Funcionalidades**

* **Resposta a Perguntas**: O assistente responde a perguntas sobre o currículo de Pablo, fornecendo informações detalhadas e relevantes com base em documentos carregados.
* **Consultas Rápidas**: Permite que os usuários escolham entre perguntas rápidas sobre o objetivo profissional, projetos realizados, habilidades e formação acadêmica de Pablo.
* **Memória de Conversa**: O assistente mantém o histórico da conversa, permitindo um contexto contínuo entre perguntas.
* **Carregamento Dinâmico de Dados**: Utiliza um arquivo PDF do currículo de Pablo para criar um índice de dados e fornecer respostas baseadas nesse conteúdo.

## **Tecnologias Utilizadas**

* **Streamlit**: Framework utilizado para criar a interface de usuário interativa.
* **LangChain**: Biblioteca usada para criar uma cadeia de conversação e integração com o modelo GPT-3.5.
* **OpenAI GPT-3.5**: Modelo de linguagem utilizado para gerar as respostas do assistente.
* **FAISS**: Biblioteca usada para indexação e busca eficiente em grandes volumes de dados (como o conteúdo do currículo em PDF).
* **Python**: Linguagem de programação utilizada no desenvolvimento do projeto.
* **HTML/CSS**: Utilizado para personalização da interface do usuário.

## **Como Executar o Projeto**

### **Requisitos**

Certifique-se de ter as seguintes dependências instaladas:

* Python 3.8 ou superior
* Bibliotecas Python:

  * `streamlit`
  * `langchain`
  * `faiss-cpu`
  * `openai`
  * `pyPDF`
  * `python-dotenv`
  * `streamlit-chat`

Você pode instalar essas dependências usando o comando após clonar este repositório:

```bash
pip install requirements.txt
```

### **Configuração de Ambiente**

Antes de rodar o assistente, você precisa configurar a chave da API da **OpenAI**. Para isso:

1. Crie uma conta na [OpenAI](https://platform.openai.com/).
2. Gere uma chave de API.
3. Adicione sua chave de API no arquivo `.env` ou nas configurações de ambiente.

Exemplo de arquivo `.env`:

```bash
OPENAI_API_KEY="sua-chave-aqui"
```

### **Execução do Projeto**

1. Após configurar a chave da API, rode o seguinte comando para iniciar o aplicativo Streamlit:

```bash
streamlit run app.py
```

2. A interface do assistente será carregada automaticamente no seu navegador.

### **Estrutura do Projeto**

```
.
├── curriculo_pdf/            # Pasta que contém o currículo PDF de Pablo
│   └── pablo_resume.pdf      # Arquivo PDF com o currículo de Pablo
├── app.py                    # Arquivo principal do aplicativo Streamlit
├── .env                      # Arquivo de configuração com chave da API OpenAI (opcional)
└── requirements.txt          # Dependências do projeto
```

### **Como Funciona**

1. **Carregamento do Currículo**: O currículo de Pablo é armazenado em formato PDF na pasta `curriculo_pdf/`. Esse arquivo é carregado e indexado usando o **FAISS**.
2. **Processamento de Perguntas**: Quando o usuário faz uma pergunta, o assistente busca no índice de documentos para fornecer respostas relevantes. Caso não haja índice, o assistente utiliza um template básico para responder.
3. **Interatividade**: O assistente possui botões de consulta rápida para facilitar a interação com perguntas frequentes sobre projetos, habilidades, formação e objetivo profissional.
4. **Memória de Conversa**: O assistente mantém o histórico de interação com o usuário, permitindo um fluxo contínuo e contextualmente relevante nas conversas.

## **Exemplo de Uso**

1. Ao abrir o aplicativo, o usuário será apresentado a um painel com informações sobre o assistente e opções de perguntas rápidas.
2. O usuário pode clicar nas perguntas rápidas ou digitar uma pergunta personalizada.
3. O assistente responde com base nas informações do currículo de Pablo ou conforme a conversa se desenvolve.



## **Licença**

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
