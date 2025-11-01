import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

# ---------------------------
# 1. Milvus + Embedding Setup
# ---------------------------
COLLECTION_NAME = "student_chandu_rag"

def setup_rag(collection_name=COLLECTION_NAME):
    """
    Create a Milvus collection, connect to it, and insert sample data.
    """

    # Connect to Milvus using alias 'rag_db'
    connections.connect("rag_db", host="localhost", port="19530")

    # Drop collection if exists (for clean setup)
    if utility.has_collection(collection_name, using="rag_db"):
        utility.drop_collection(collection_name, using="rag_db")

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    ]

    schema = CollectionSchema(fields, description="Employee policy documents")
    collection = Collection(collection_name, schema, using="rag_db")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Sample data
    documents = [
        "Performance reviews are conducted annually in December.",
        "Employees must submit timesheets by Friday at 5 PM each pay period.",
        "Overtime must be approved in advance by a supervisor.",
        "Paid holidays include all federal holidays and one floating holiday per year.",
        "Sick leave accrues at one day per month of active employment.",
        "Employees are eligible for a 401(k) plan after six months of service.",
        "Company laptops and equipment must be returned upon termination of employment.",
        "Dress code is business casual from Monday to Thursday, and casual on Fridays.",
        "Employees must notify their manager at least two weeks in advance for planned time off.",
        "Training and development reimbursements require prior written approval.",
        "Employees must badge in and out when entering or leaving company premises.",
        "Company email and internet use are monitored for compliance with IT policy.",
        "All meetings should start and end within the scheduled time slot.",
        "Expense reports must be submitted within 10 days of travel completion.",
        "Work hours are from 9:00 AM to 5:00 PM, Monday through Friday.",
        "Team meetings are held weekly to review project progress and priorities.",
        "Employees are paid twice a month.",
        "Confidential company information must not be shared outside the organization.",
        "Employees may request ergonomic equipment through the HR department.",
        "Resignation notice must be provided in writing at least two weeks prior to departure."
    ]

    embeddings = model.encode(documents).tolist()

    # Insert data into Milvus
    collection.insert([documents, embeddings])
    collection.flush()

    # Create index
    collection.create_index(
        "embedding",
        {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    )
    collection.load()

    print(f"RAG setup complete with {len(documents)} sample records in '{collection_name}'.")
    print( )
    return collection_name


# ---------------------------
# 2. Setup the Data in the Vector DB
# ---------------------------
if __name__ == "__main__":
    setup_rag()
    # Load collection
    collection = Collection(COLLECTION_NAME, using="rag_db")

    # Query all records (limit = small number to avoid large dumps)
    results = collection.query(expr="id >= 0", output_fields=["id", "content"], limit=10)
    
    print(f"number of entities:{collection.num_entities}")
    
    print(f"Retrieved {len(results)} records from collection '{COLLECTION_NAME}':\n")
    for r in results:
        print(f"ID: {r['id']} | Content: {r['content']}")