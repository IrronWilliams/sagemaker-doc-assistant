import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv() # Loads variables from .env file

'''
Checking: Program chunks the Sage Developer PDF into 1000 character chunks with 200 character overlaps
Embedding: Then the chunks are sent to OpenAIEmbeddings (model="text-embedding-3-small") to create embeddings
Storing: Tne embeddings are stored in a Chroma vector database in the /chroma_db folder

Confirms donwload script has run and Sage Developer Guide is present in Docs folder. 

PyPDFLoader -> opens the bindary PDF file and extracts all text. 
loader.load() -> returns a list of Document objects. Each item in the list is a LangChain Document object 
representing a single page of the PDF (including metadata like page number).

text_splitter -> initialize splitter and and tell it I want chunks of appx 1k chars. 
The overlap=200 ensures no context or meaning is lost at borders where text is sliced. 
For example, if Chunk A ends with "the dog" and "sat on" and Chunk B starts with "the mat", 
overlap 200 tells the splitter to keep 200 chars from the bottom of one chunk and paste it at the top of the next. 
Basically if Chunk A ends in middle of a sentence, Chunk B will start 200 chars back in Chunk A. 

split_documents() -> performs the slicing, turning the ~9500 pages into thousands of overlapping chunks. 

Chroma.from_documents -> performs the heavy lifting. Takes the thousands of chunks and passes them to OpenAIEmbedding
using modern text-embedding-3-small model. OpenAI converts and send back the text chunks into vectors (lists of numbers representing contextual meaning). 

persist_directory=DB_DIR -> tells Chroma to save the vectors (and orignal text) database to the local folder "chroma_db". (instead of keeping them in RAM for current session only)
'''

PDF_PATH = "Docs/sagemaker-dg.pdf"
DB_DIR = "chroma_db"

def ingest_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF not found at {PDF_PATH}. Please run download_pdf.py first.")
        return

    print("Loading PDF (this may take a moment for a large document)...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    
    print("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print(f"Creating Chroma vector database with {len(splits)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=DB_DIR
    )
    
    print(f"Database successfully created and saved to {DB_DIR}")

if __name__ == "__main__":
    ingest_pdf()
