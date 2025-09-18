from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from . import retrieval, generation

app = FastAPI()
router = APIRouter()

vectorstore = retrieval.load_vectorstore()
qa_chain = generation.create_qa_chain(vectorstore)

class QueryRequest(BaseModel):
    query: str

@router.get("/retrieve")
def retrieve_documents(query: str):
    retriever = retrieval.get_retriever(vectorstore)  # Make sure to pass vectorstore if needed
    docs = retriever.get_relevant_documents(query)
    results = [doc.page_content for doc in docs]
    return {"query": query, "results": results}

@router.post("/generate")
async def generate_answers(request: QueryRequest):
    from starlette.concurrency import run_in_threadpool
    answer = await run_in_threadpool(generation.generate_answer, request.query, qa_chain)
    return {"results": [answer]}

app.include_router(router)



