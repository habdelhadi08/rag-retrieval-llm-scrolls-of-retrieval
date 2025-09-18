from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.document_loaders import WikipediaLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
import time
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)

def profile_time(func):
    """Decorator to measure execution time of functions."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"{func.__name__} took {elapsed:.3f} seconds")
        return result
    return wrapper

@profile_time
def create_vectorstore_from_wikipedia(
    queries=None,
    lang="en",
    index_path="faiss_index"
):
    if queries is None:
        queries = [
            "Machine learning",
            "Artificial intelligence",
            "Deep learning",
            "Supervised learning",
            "Unsupervised learning",
            "Overfitting"
        ]

    docs = []
    for topic in queries:
        loader = WikipediaLoader(query=topic, lang=lang, load_max_docs=2)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    vectorstore.save_local(index_path)
    logging.info(f"Vectorstore created and saved to: {index_path}")
    return vectorstore

@profile_time
def load_vectorstore(
    index_path="faiss_index"
):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    return vectorstore

@profile_time
def create_qa_chain(vectorstore, k=5):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    pipe = pipeline("text2text-generation", model="google/flan-t5-small")
    llm = HuggingFacePipeline(pipeline=pipe)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

# Simple LRU cache to store answers for repeated queries
@lru_cache(maxsize=128)
@profile_time
def generate_answer_cached(query, qa_chain):
    # We clear chat history for caching simplicity, could be improved
    result = qa_chain({"question": query, "chat_history": []})
    answer = result.get("answer", "No answer found.")
    return answer

def generate_answer(query, qa_chain, chat_history):
    answer = generate_answer_cached(query, qa_chain)
    chat_history.append((query, answer))
    return answer


if __name__ == "__main__":
    # Uncomment to create and save the vectorstore initially
    # vectorstore = create_vectorstore_from_wikipedia()

    vectorstore = load_vectorstore()

    qa_chain = create_qa_chain(vectorstore, k=5)

    chat_history = []

    questions = [
        "Define artificial intelligence",
        "What is machine learning?",
        "Explain overfitting in machine learning"
    ]

    for i, question in enumerate(questions, 1):
        logging.info(f"Q{i}: {question}")
        answer = generate_answer(question, qa_chain, chat_history)
        logging.info(f"A{i}: {answer}\n")

    logging.info("Chat History:")
    for q, a in chat_history:
        logging.info(f"Q: {q} -> A: {a}")

