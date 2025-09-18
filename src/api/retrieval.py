import time
import logging
from functools import lru_cache
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
INDEX_PATH = "faiss_index"

def profile_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"{func.__name__} took {elapsed:.3f} seconds")
        return result
    return wrapper

@lru_cache(maxsize=1)
@profile_time
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
@profile_time
def load_vectorstore(index_path=INDEX_PATH):
    embedding_model = load_embedding_model()
    vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    return vectorstore

@profile_time
def create_vectorstore_from_wikipedia(
    queries=None,
    lang="en",
    index_path=INDEX_PATH
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embedding_model = load_embedding_model()
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    vectorstore.save_local(index_path)
    logging.info(f"Vectorstore created and saved to: {index_path}")
    return vectorstore

@lru_cache(maxsize=1)
@profile_time
def get_retriever(k=5):
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

@profile_time
def query_retriever(query, k=5):
    retriever = get_retriever(k)
    docs = retriever.get_relevant_documents(query)
    return docs

if __name__ == "__main__":
    # Uncomment to (re)create vectorstore initially (slow step)
    # create_vectorstore_from_wikipedia()

    # Load vectorstore and retriever (cached, fast after first load)
    vectorstore = load_vectorstore()
    retriever = get_retriever(k=5)

    # Test query with profiling
    question = "What is overfitting?"
    docs = query_retriever(question, k=5)

    print(f"Retrieved {len(docs)} documents for query '{question}':")
    for i, doc in enumerate(docs, 1):
        print(f"--- Document chunk {i} ---")
        print(doc.page_content[:500])  # Print first 500 characters


