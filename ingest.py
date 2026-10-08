import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def main():
    print("Iniciando o processo de ingestão de documentos (PDFs)...")
    
    docs_dir = "./docs"
    persist_dir = "./chroma_db"
    
    if not os.path.exists(docs_dir):
        print(f"Erro: O diretório '{docs_dir}' não foi encontrado.")
        print("Por favor, crie a pasta 'docs' e adicione os arquivos PDF dentro dela antes de rodar a ingestão.")
        return

    # 1. Carregar documentos PDF
    print(f"Lendo arquivos PDF do diretório: {docs_dir}")
    try:
        loader = PyPDFDirectoryLoader(docs_dir)
        documents = loader.load()
        if not documents:
            print("Nenhum documento encontrado na pasta docs.")
            return
        print(f"Total de páginas carregadas: {len(documents)}")
    except Exception as e:
        print(f"Erro ao carregar documentos: {e}")
        return

    # 2. Dividir em chunks (Pedaços menores)
    print("Dividindo documentos em chunks de texto...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total de chunks gerados: {len(chunks)}")

    # 3. Gerar Embeddings usando Google Gemini e salvar no ChromaDB
    print("Gerando embeddings e populando o ChromaDB...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="projetofinal_rag_docs"
        )
        print("\n[SUCESSO] Ingestão finalizada e banco vetorial criado!")
    except Exception as e:
        print(f"Erro durante a criação dos embeddings/banco vetorial: {e}")

if __name__ == "__main__":
    main()
