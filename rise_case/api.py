import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import pinecone
import tempfile
import openai
import os
from pydantic import BaseModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# swagger api:
# http://0.0.0.0:8000/docs#


os.environ["OPENAI_API_KEY"] = "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"

app = FastAPI()

# Pinecone configuration
PINECONE_INDEX_NAME = 'chatbot'
PINECONE_NAME_SPACE = 'gcp-starter'
MODEL_NAME = "gpt-4"


# This function is used to upload a PDF file and load its embeddings into Pinecone.
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global index
    try:
        # Set the OpenAI API key.
        openai.api_key = os.environ["OPENAI_API_KEY"]

        # Create a temporary directory to store the uploaded PDF file.
        upload_dir = tempfile.mkdtemp()

        # Get the path to the uploaded PDF file.
        pdf_path = os.path.join(upload_dir, file.filename)

        # Write the contents of the uploaded PDF file to the file.
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(file.file.read())

        # Load the PDF file into a list of raw documents.
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()

        # Split the raw documents into smaller chunks.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(raw_docs)

        # Generate embeddings for the text chunks.
        embeddings = OpenAIEmbeddings()

        # Print a message to the console indicating that the vector store is being created.
        print('Creating vector store...')

        # Initialize the Pinecone client with the specified API key and environment.
        pinecone.init(api_key="df65b420-76ae-4d82-badf-3c4f1bd19457", environment=PINECONE_NAME_SPACE)

        # Create a Pinecone index from the text chunks and embeddings.
        index = Pinecone.from_documents(docs, embeddings, index_name=PINECONE_INDEX_NAME)

        # Check if the Pinecone index is empty.
        if index is None:
            # Log an error message if the Pinecone index is empty.
            logging.error("Pinecone from_documents index is empty")
            # Raise an exception if the Pinecone index is empty.
            raise Exception

        # Return a message indicating that the PDF was uploaded and its embeddings were loaded into Pinecone successfully.
        return {"message": "PDF uploaded and embeddings loaded into pinecone successfully"}

    except Exception as error:
        # Print the error message to the console.
        print('Error', error)
        # Raise an HTTPException with status code 500 and the error message as the detail.
        raise HTTPException(status_code=500, detail="Failed to process PDF")


class Question(BaseModel):
    # This class defines the schema for the question payload.
    question: str


@app.post("/ask_question/")
async def ask_question(question_data: Question):
    # This function is used to ask a question and get the answer.
    global index
    try:
        # Check if there is a document loaded.
        if index is None:
            # Log an error message and raise an exception.
            logging.error("There is no document loaded. Please ue /upload/ endpoint firstly:)")
            raise Exception

        # Get the user's question.
        user_question = question_data.question

        # Get the most similar documents to the user's question.
        similar_docs = get_similiar_docs(index, user_question)

        # Initialize the ChatGPT-3 model.
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)

        # Load the question-answering chain.
        chain = load_qa_chain(llm, chain_type="stuff")

        # Get the answer from the question-answering chain.
        answer = chain.run(input_documents=similar_docs, question=user_question)

        # Return the answer.
        response = {"answer": answer}
        return response

    except Exception as error:
        # Log the error message.
        print('Error', error)

        # Raise an HTTPException with status code 500 and the error message as the detail.
        raise HTTPException(status_code=500, detail="Failed to response to question")


def get_similiar_docs(index, query, k=2, score=False):
    # This function gets the most similar documents to the given query.

    # Check if the score flag is set.
    if score:
        # Get the documents with scores.
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        # Get the documents without scores.
        similar_docs = index.similarity_search(query, k=k)

    # Return the similar documents.
    return similar_docs


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
