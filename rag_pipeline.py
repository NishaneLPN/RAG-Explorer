import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import  OpenAIEmbeddings, ChatOpenAI  
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.stores import InMemoryStore
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

def load_and_chunk(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    return documents, chunks

def build_vectorstore(chunks, persist_directory="./chroma_db"):
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore, embeddings

def get_retriever(retriever_type, vectorstore, documents, embeddings):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    if retriever_type == "Similarity":
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
    
    elif retriever_type == "MMR":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
        )
    
    elif retriever_type == "Parent Document":
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
        docstore = InMemoryStore()
        child_vectorstore = Chroma(
            collection_name="child_chunks",
            embedding_function=embeddings
        )
        retriever = ParentDocumentRetriever(
            vectorstore=child_vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        retriever.add_documents(documents)
        return retriever
    
    elif retriever_type == "Multi Query":
        custom_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Generate 3 different versions of the given question 
            from completely different angles:
            - Version 1: Definition or characteristics
            - Version 2: Underlying mechanism or process  
            - Version 3: Consequences or limitations
            
            Original question: {question}
            Provide exactly 3 questions, one per line, no numbering:"""
        )
        return MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            llm=llm,
            prompt=custom_prompt
        )

def get_answer(question, retriever):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    response = qa_chain.invoke({"query": question})
    return response["result"], response["source_documents"]