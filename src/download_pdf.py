import requests
import os

'''
This is the primary source document for the app. 
Using requests and os libraries, program downloads the large ~9500 page AWS SageMaker Developer Guide
and saves in the Docs folder. 

'''


PDF_URL = "https://docs.aws.amazon.com/pdfs/sagemaker/latest/dg/sagemaker-dg.pdf"
SAVE_PATH = "Docs/sagemaker-dg.pdf"

def download_pdf():
    # Ensure the Docs directory exists
    os.makedirs("Docs", exist_ok=True) #exist_ok True prevents FileExistsError exception if folder already exists
    
    print(f"Downloading SageMaker Developer Guide from {PDF_URL}...")
    response = requests.get(PDF_URL, stream=True) # stream=True prevents loading the 10k file entirely into memory at once (likely crash sys). Stream keeps memory footprints small. 
    response.raise_for_status() #checks for HTTP errors (like 404, 500) and raises an exception if found. Prevents saving incomplete or corrupted file. 
    
    with open(SAVE_PATH, "wb") as f: #PDFs are binary files. Instruct Pyton to write binary data with 'wb'
        for chunk in response.iter_content(chunk_size=8192): #safely download by pulling and writting 8192 bytes at a time. Removes bytes from RAM, then grabs next 8192 chunk until download complete.
            f.write(chunk)
            
    print(f"Successfully downloaded to {SAVE_PATH}")

if __name__ == "__main__":
    download_pdf()
