import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma  # Updated import
import chromadb
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the GROQ API key environment variable
os.environ['GROQ_API_KEY'] = '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'

# Initialize the Hugging Face Embedding Model (using Sentence Transformers)
model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
embedding_model = SentenceTransformer(model_name)  # You can choose any other model from HuggingFace
logger.info(f"Using embedding model: {model_name}")

# Custom embedding function class
class CustomEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        return self.model.encode([query])[0]

    def embed_documents(self, documents):
        return self.model.encode(documents)

# Initialize the custom embedding function
embedding_function = CustomEmbeddingFunction(embedding_model)

# Initialize Chroma Client
client = chromadb.Client()

# Specify the path to your CSV file
file_path = os.getenv('CSV_FILE_PATH', '/home/abdulsamad/blash_project/Procore_Subcontractor_Invoice_20_Duplicates.csv')
logger.info(f"Loading data from CSV file: {file_path}")

# Load the CSV data using the CSVLoader
try:
    loader = CSVLoader(file_path=file_path)
    logger.info("Loading documents...")
    data = loader.load()
    logger.info(f"Number of documents loaded: {21}")
except Exception as e:
    logger.error(f"Error loading CSV file: {e}")
    raise

# Convert each document's content to embeddings and store them in Chroma
texts = [doc.page_content for doc in data]
metadatas = [doc.metadata for doc in data]

# Create or connect to a Chroma collection
collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'mlb_teams_2012')
collection = client.create_collection(name=collection_name)
logger.info(f"Created or connected to Chroma collection: {collection_name}")

# Add embeddings to Chroma
try:
    embeddings = embedding_function.embed_documents(texts)
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=[str(i) for i in range(len(texts))],
        embeddings=embeddings
    )
    logger.info("Documents embedded and stored in ChromaDB successfully.")
except Exception as e:
    logger.error(f"Error during embedding or storing documents: {e}")
    raise

# Set up a retrieval mechanism to query the stored embeddings
logger.info("Setting up retrieval...")
# Initialize the retriever with the custom embedding function
retriever = Chroma(
    collection_name=collection_name,
    client=client,
    embedding_function=embedding_function
).as_retriever(search_kwargs={"k": 3})

# Initialize ChatOpenAI model for conversational queries with custom settings
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ['GROQ_API_KEY'],
    model_name="llama3-8b-8192",
    temperature=0,
    #max_tokens=1000,
)

# Initialize the ConversationalRetrievalChain with the custom LLM
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever
)

# Perform a sample query
query = "Which team had the highest payroll in 2012?"
chat_history = []  # Initialize an empty chat history
logger.info(f"Query: {query}")

try:
    response = qa_chain.invoke({"question": query, "chat_history": chat_history})
    logger.info(f"Answer: {response}")
except Exception as e:
    logger.error(f"Error during query execution: {e}")
    raise
