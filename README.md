# RAG - Chatbot Especialista 🤖

Este repositório contém o código de um Projeto que implementa um assistente conversacional utilizando a arquitetura **RAG (Retrieval-Augmented Generation)**. O objetivo é permitir que o chatbot responda perguntas de forma precisa e baseada em documentos PDF fornecidos ao banco de dados, sem depender exclusivamente do conhecimento pré-treinado do LLM.

## 🛠 Tecnologias Utilizadas
- **Python**: Linguagem principal.
- **Streamlit**: Para a criação rápida de uma interface web interativa.
- **LangChain**: Framework para orquestrar LLMs, manipulação de prompts e cadeias de RAG.
- **ChromaDB**: Banco de dados vetorial local para armazenar e buscar os embeddings.
- **Google Gemini**: API de LLM (gemini-1.5-flash) para geração das respostas e embeddings.

## 🚀 Como executar este projeto localmente

### 1. Pré-requisitos
Certifique-se de que você tem o Python instalado em sua máquina. Para clonar o repositório e preparar o ambiente:

```bash
git clone https://github.com/SeuUsuario/SeuRepositorio.git
cd SeuRepositorio
python -m venv venv
# Ativar venv no Windows: venv\Scripts\activate
# Ativar venv no Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar a Chave de API
O projeto utiliza a API do Google Gemini. Copie o arquivo de exemplo de ambiente:

```bash
# Crie um arquivo .env na raiz do projeto contendo:
GEMINI_API_KEY=sua_chave_de_api_do_google_aqui
```

### 3. Preparando o Banco de Dados Vetorial
Crie uma pasta chamada `docs` na raiz do repositório e coloque todos os seus documentos em formato PDF dentro dela. Em seguida, execute o script de ingestão para converter os PDFs em blocos de texto e gerar os embeddings (vetores matemáticos) no banco ChromaDB.

```bash
python ingest.py
```
*(Após rodar esse comando, você verá que uma pasta `chroma_db` será criada com os dados salvos com sucesso).*

### 4. Executando o Chatbot (App)
Com o banco populado, inicie o servidor Streamlit:

```bash
streamlit run app.py
```
Pronto! Uma janela será aberta em seu navegador. Faça perguntas específicas sobre o assunto dos PDFs e o bot irá retornar a resposta junto com a transparência de qual página e documento a informação foi retirada!
