import cohere
import chromadb
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import CohereEmbeddings
from config import COHERE_API_KEY

# Cohere client initialization
cohere_client = cohere.Client(COHERE_API_KEY)

# Load CSV file using LangChain's CSVLoader
loader = CSVLoader(file_path='/home/abdulsamad/blaash_assignement/Procore_Subcontractor_Invoice_20_Duplicates.csv')
data = loader.load()

# Function to generate embeddings using Cohere API
def generate_embeddings(texts):
    response = cohere_client.embed(texts=texts, model="multilingual-1.0")
    return response.embeddings

# Generate embeddings for each document in CSV
texts = [doc.page_content for doc in data]
embeddings = generate_embeddings(texts)

# Chroma client initialization and vector store creation
client = chromadb.Client()
collection_name = "mlb_teams_2012"
collection = client.create_collection(collection_name)

# Add embeddings and documents to the ChromaDB collection
for idx, (embedding, doc) in enumerate(zip(embeddings, data)):
    collection.add(
        embeddings=[embedding],
        metadatas=[doc.metadata],
        documents=[doc.page_content],
        ids=[str(idx)]
    )

print(f"Added {len(embeddings)} embeddings to the ChromaDB collection.")
