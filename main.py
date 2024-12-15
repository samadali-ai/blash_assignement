# main.py
from load_data import load_csv
from convert_to_documents import convert_rows_to_documents
from text_splitter import split_documents
from embedding import generate_embeddings_and_store
from qa_chain import setup_qa_chain, run_qa_chain

def main():
    # Step 1: Load data
    df = load_csv('data.csv')
    if df is None:
        return

    # Step 2: Convert rows to documents
    documents = convert_rows_to_documents(df)

    # Step 3: Optionally, split documents into smaller chunks
    split_docs = split_documents(documents)

    # Step 4: Generate embeddings and store in Chroma
    vector_store = generate_embeddings_and_store(split_docs)

    # Step 5: Set up the QA chain
    qa_chain = setup_qa_chain(vector_store)

    # Step 6: Run a query through the QA chain
    query = "What is the vendor name for the contract SC-001?"
    run_qa_chain(qa_chain, query)

if __name__ == "__main__":
    main()
