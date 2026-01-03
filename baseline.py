# baseline.py (단일 RAG 에이전트)
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# 1. 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

if os.path.exists("./chroma_db"):
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="baby_knowledge"
    )
else:
    raise ValueError("DB 없음")

# 2. 단순 검색 및 답변 체인
def simple_rag_answer(question: str):
    # (1) 검색 (구분 없이 그냥 검색)
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])
    
    # (2) 단순 답변 (전문가 페르소나 없음)
    prompt = ChatPromptTemplate.from_template(
        "당신은 AI 어시스턴트입니다. 아래 정보를 바탕으로 질문에 답하세요:\n\n"
        "정보: {context}\n\n"
        "질문: {question}"
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

if __name__ == "__main__":
    print(simple_rag_answer("아기가 밤에 안자요"))