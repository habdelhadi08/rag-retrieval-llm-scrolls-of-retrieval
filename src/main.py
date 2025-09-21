import redis
from src.config.config import REDIS_HOST, REDIS_PORT
import json
import time
import logging
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, Header
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session, selectinload
from fastapi import Query
from src.database.database import Base, engine, get_db
from src.models import ChatHistory
from src.api.generation import load_vectorstore, create_qa_chain

logging.basicConfig(level=logging.INFO)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
API_KEY = "mysecretkey123"

# Connect to Redis (adjust host/port if needed)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

async def verify_api_key(api_key: str = Query(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API Key")

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

vectorstore = load_vectorstore()
qa_chain = create_qa_chain(vectorstore)

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request, db: Session = Depends(get_db), api_key: str = Depends(verify_api_key)):
    chat_history = db.query(ChatHistory)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "chat_history": [(item.question, item.answer) for item in chat_history]
    })

@app.post("/answer", response_class=HTMLResponse)
async def answer_question(request: Request, question: str = Form(...), db: Session = Depends(get_db), api_key: str = Depends(verify_api_key)):
    # Check Redis cache first
    cached_answer = redis_client.get(question)
    if cached_answer:
        logging.info("Answer fetched from Redis cache.")
        answer = cached_answer
    else:
        start_time = time.time()
        result = qa_chain.invoke({"question": question, "chat_history": []})
        answer = result.get("answer", "No answer found.")
        elapsed = time.time() - start_time
        logging.info(f"QA chain invoke took {elapsed:.3f} seconds")

        # Save answer to Redis cache with TTL of 1 hour (3600 seconds)
        redis_client.set(question, answer, ex=3600)

        # Save to DB
        chat = ChatHistory(question=question, answer=answer)
        db.add(chat)
        db.commit()

    chat_history = db.query(ChatHistory).options(selectinload()).limit(100).all()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "answer": answer,
        "chat_history": [(item.question, item.answer) for item in chat_history],
        "question": question
    })




