from datasets import load_dataset

# Load Wealth Alpaca dataset
wealth_alpaca_dataset = load_dataset("gbharti/wealth-alpaca_lora")
wealth_alpaca_df = pd.DataFrame(wealth_alpaca_dataset["train"])

# Standardize columns
financial_qa = financial_qa.rename(columns={"question": "instruction", "answer": "output"})
financial_qa["input"] = ""  # Add empty 'input' column

# Combine both datasets
combined_dataset = pd.concat([financial_qa, wealth_alpaca_df], ignore_index=True)

# Save combined dataset if needed
combined_dataset.to_csv('combined_dataset.csv', index=False)

# Step 4: Generate Embeddings using MiniLM
embedder = SentenceTransformer("all-MiniLM-L6-v2")
combined_dataset["combined_text"] = combined_dataset["instruction"] + " " + combined_dataset["input"]
corpus = combined_dataset["combined_text"].tolist()
embeddings = embedder.encode(corpus)


# Clean the corpus by removing invalid entries (e.g., NaN or None)
corpus = [doc if isinstance(doc, str) else "" for doc in corpus]

# Verify there are no invalid documents
print(f"Total documents after cleaning: {len(corpus)}")

def store_in_chromadb(corpus, embeddings):
    """Store documents and embeddings in ChromaDB."""
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Create or get collection
    collection = chroma_client.get_or_create_collection(name="financial_wealth_corpus")

    # Convert embeddings to list if they're numpy array
    if hasattr(embeddings, 'tolist'):
        embeddings = embeddings.tolist()

    # Add documents in batches
    batch_size = 64
    total_docs = len(corpus)

    print(f"Starting to store {total_docs} documents...")

    try:
        for i in range(0, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            batch_docs = corpus[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            batch_ids = [str(j) for j in range(i, batch_end)]
            batch_metadata = [{"id": j} for j in range(i, batch_end)]

            print(f"Processing batch {i//batch_size + 1}: documents {i} to {batch_end}")

            # Verify data shapes before adding
            print(f"Batch size: {len(batch_docs)} documents, {len(batch_embeddings)} embeddings")

            collection.add(
                documents=batch_docs,
                embeddings=batch_embeddings,
                ids=batch_ids,
                metadatas=batch_metadata
            )

            print(f"Successfully added batch {i//batch_size + 1}")

        # Verify the collection has data
        collection_count = collection.count()
        print(f"Total documents in collection: {collection_count}")

        return collection

    except Exception as e:
        print(f"Error storing documents: {str(e)}")
        print(f"Document example: {corpus[0] if corpus else 'No documents'}")
        print(f"Embedding example shape: {len(embeddings[0]) if embeddings else 'No embeddings'}")
        raise

# Test the storage
def verify_chromadb_storage():
    print("\nVerifying ChromaDB storage:")
    client = chromadb.PersistentClient(path="./chroma_db")
    collections = client.list_collections()
    print(f"Number of collections: {len(collections)}")
    for collection in collections:
        print(f"Collection name: {collection.name}")
        print(f"Collection count: {collection.count()}")


store_in_chromadb(corpus, embeddings)
verify_chromadb_storage()
