import streamlit as st
import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG - Artigos Científicos", page_icon="🤖", layout="wide")

load_dotenv()

@st.cache_resource
def load_rag_components():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="projetofinal_rag_docs"
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    
    prompt = ChatPromptTemplate.from_template(
        "Você é um assistente especialista na leitura e análise dos artigos científicos fornecidos.\n"
        "Use EXCLUSIVAMENTE o contexto abaixo para responder à pergunta.\n"
        "Se não souber a resposta baseada no contexto, diga claramente que a informação não consta nos artigos.\n\n"
        "Contexto:\n{contexto}\n\n"
        "Pergunta: {pergunta}\n\n"
        "Responda de forma clara, profissional e SEMPRE cite a fonte e a página no final."
    )
    return vectordb, llm, prompt

try:
    vectordb, llm, qa_prompt = load_rag_components()
except Exception as e:
    st.error(f"Erro ao carregar os dados. Você já rodou o arquivo 'ingest.py'? Detalhes: {e}")
    st.stop()

st.title("🤖 RAG Chatbot: Base de Conhecimento Científica")
st.markdown("Bem-vindo! Tire dúvidas consultando automaticamente nossa base de artigos em PDF sobre Deepfakes e Blockchain.")

with st.sidebar:
    st.header("⚙️ Configurações da Busca")
    top_k = st.slider(
        "Quantidade de blocos para contexto (Top K)", 
        min_value=1, 
        max_value=10, 
        value=4,
        help="Quantos trechos de PDFs o sistema vai ler para tentar te responder."
    )
    st.markdown("---")
    st.markdown("**Desenvolvido com:** LangChain + Gemini + Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua pergunta sobre os artigos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Lendo artigos e processando a resposta..."):
            retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(prompt)
            
            contexto_blocos = []
            fontes_usadas = set()
            for d in docs:
                source_path = d.metadata.get("source", "Desconhecido")
                nome_arquivo = os.path.basename(source_path)
                page = d.metadata.get("page", "?")
                
                contexto_blocos.append(f"[Fonte: {nome_arquivo} | Pág: {page}]\n{d.page_content}")
                fontes_usadas.add(f"📄 **{nome_arquivo}** (Pág. {page})")
            
            contexto_formatado = "\n\n---\n\n".join(contexto_blocos)
            
            msg = qa_prompt.format_messages(contexto=contexto_formatado, pergunta=prompt)
            resposta = llm.invoke(msg).content
            
            st.markdown(resposta)
            
            with st.expander("🔍 Transparência: Fontes Consultadas"):
                st.write("**Trechos recuperados dos PDFs pelo modelo:**")
                for fonte in fontes_usadas:
                    st.write(fonte)
                st.write("---")
                st.info(contexto_formatado)
            
    st.session_state.messages.append({"role": "assistant", "content": resposta})
