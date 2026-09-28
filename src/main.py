import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

'''
Program is the backbone of the RAG application.

Pydantic model tells FastAPI to expect JSON object with a key called "question". 
FastAPI will reject any request that doesn't have this key and format, preventing 
bad data from ever reaching the AI and rejecting request before code runs. 

Initialize the vecotr db connection/RAG components globally. Initialize connection 
outside of the chat_endpoint() function to allow server to load the DB only once (at startup). 
This prevents continuous reconnection every time a user asks a question. 

search_kwargs={"k": 3} -> tells retirever to return the top 3 most relevant 1000 chars chunks from DB. 

The prompt includes instruction for LLM to say I Don't Know if answer not found in context.
This helps reduce hallucinations. 

{context} -> placeholders the LCEL pipeline will fill automatically.
{question} -> placeholders the LCEL pipeline will fill automatically.


@app-post is a decorator that tells FastAPI to run the chat_endpoint() function when a POST request is made to the /chat endpoint.

async def allows server to handle multiple users asking questions at the same time without getting stuck on one request.

LCEL Pipeline: LangChain Expression Language (LCEL) is used to create the RAG pipeline. Reads from top to bottom like assembly line. 

RunnablePassthrough() -> Takes user's raw string and drops directly into the {question} placeholder. 
Simultaneously, the retriever takes the user's question, searches Chroma, fetches the 3 best chunks and 
drops them into the {context} placeholder.

prompt_template -> Takes the fully populated context and question and formats them into a final string. 

llm -> Passes the massive string ove the Internet to OpenAI.

StrOutputParser() -> Interceepts messy JSON response that includes metadata and strips the answer from the response object, 
returning only the raw text string in a clean readable text for the user.
 
Wrap the invoke() method in a try/except block to catch any errors during processing and prevent server crash.
The invoke() method executes the entire RAG pipeline from start to finish.
Return the answer as a Python dict. FastAPI will automaticallly convert the dict to JSON and send back to user browser (or Streamlit).

Utilizing unicorn to boot the local web server. 
hot reloading with reload=True is enabled to automatically restart the server when code changes are detected.

'''



DB_DIR = "chroma_db"

# Data Model for FastAPI
class ChatRequest(BaseModel):
    question: str

app = FastAPI(title="SageMaker AI Assistant") #create web server object

# Vector DB Connection
if os.path.exists(DB_DIR):
    vectorstore = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # retrieve top 3 most relevant chunks
else:
    retriever = None
    print("Warning: Chroma DB not found. Please run ingestion.py first.")

llm = ChatOpenAI(model="gpt-4o-mini") # Initialize the LLM, specific generation model

# Define the Prompt
prompt_template = ChatPromptTemplate.from_template(
    """
    You are an expert Amazon SageMaker assistant. Answer the question based ONLY on the provided context.
    If the answer is not in the context, say "I don't know based on the provided documentation."
    
    Context: {context} 
    
    Question: {question} 
    """
)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not retriever:
        raise HTTPException(status_code=500, detail="Vector database not initialized. Run ingestion.py first.")
        
    # LangChain Expression Language (LCEL) Pipeline
    rag_pipeline = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    try:
        response = rag_pipeline.invoke(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI server locally on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
