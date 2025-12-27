import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. 환경 변수 로드 (API Key 필요)
load_dotenv()

# API Key 확인
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ Error: .env 파일에 GOOGLE_API_KEY가 없습니다.")
    exit()

def ingest_data():
    print("🚀 BabySquad 지식 주입(공부) 시작...")

    # ---------------------------------------------------------
    # [1] 문서 로드 (Load Documents)
    # data 폴더에 있는 txt, pdf 파일을 모두 읽어옵니다.
    # ---------------------------------------------------------
    data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"⚠️ '{data_path}' 폴더가 없어서 생성했습니다. 학습할 파일을 넣어주세요!")
        return

    # 텍스트 파일 로더
    loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # PDF 파일이 있다면 아래 주석 해제
    pdf_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    if not documents:
        print("📂 data 폴더가 비어있습니다. 학습할 텍스트 파일(.txt)을 넣어주세요.")
        return

    print(f"📚 총 {len(documents)}개의 문서를 읽어왔습니다.")

    # ---------------------------------------------------------
    # [2] 문서 분할 (Split Documents)
    # 책을 한 번에 다 외울 수 없으니, 문단 단위로 쪼갭니다.
    # ---------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 1000자 단위로 자름
        chunk_overlap=200 # 문맥이 끊기지 않게 200자씩 겹치게 자름
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  문서를 {len(chunks)}개의 조각(Chunk)으로 잘랐습니다.")

    # ---------------------------------------------------------
    # [3] 임베딩 및 DB 저장 (Embed & Store)
    # 텍스트를 AI가 이해하는 숫자(Vector)로 바꿔서 ChromaDB에 저장합니다.
    # ---------------------------------------------------------
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # DB 저장 경로
    persist_directory = "./chroma_db"
    
    print("💾 데이터베이스에 저장 중... (시간이 조금 걸릴 수 있습니다)")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="baby_knowledge" # 데이터베이스 이름
    )
    
    print(f"✅ 학습 완료! 데이터가 '{persist_directory}'에 저장되었습니다.")

if __name__ == "__main__":
    ingest_data()