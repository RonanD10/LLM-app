from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
import faiss
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=5,
    api_key=os.getenv("OPENAI_API_KEY"),
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def create_vector_db():
    
    loader = CSVLoader(file_path='Heart_Lung_and_BloodQA.csv', source_column="Question")
    data = loader.load()

    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    
    return vectordb 


vectordb = create_vector_db()

def get_qa_chain():

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "answer" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain


if __name__ == "__main__":
    chain = get_qa_chain()
    # print(chain("How common is asthma?")['result'])
