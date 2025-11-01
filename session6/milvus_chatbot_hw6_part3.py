import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import json

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)


# ---------------------------
# Semantic Search Function
# ---------------------------

def retrieve_similiar_contexts(query, collection_name="student_chandu_rag", top_k=3):
    """
    Given a user query, return top K semantically similar texts from Milvus.
    """
    connections.connect("default", host="localhost", port="19530")

    collection = Collection(collection_name)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_vector = model.encode([query]).tolist()

    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["content"]
    )

    top_docs = []
    for hit in results[0]:
        top_docs.append({
            "content": hit.entity.get("content"),
            "score": hit.distance
        })

    return top_docs


# ---------------------------
# LLM Answer Generation
# ---------------------------

def generate_answer(query, contexts):
    """
    Generate an answer using OpenAI GPT model based on retrieved contexts.
    """
    load_dotenv(override=True, dotenv_path="../.env")
    my_api_key = os.getenv("OPEN_AI_API_KEY")

    client = OpenAI(api_key=my_api_key)


    context_str = "\n".join(contexts)
    print(f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:")
    prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-5-nano",  # or gpt-4o if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based only on context."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()


#---------------------------
#Combined RAG Query Pipeline
#---------------------------

def retrieve_and_generate_response(query):
     """
     Perform full RAG flow: search + LLM answer generation.
     """
     top_docs = retrieve_similiar_contexts(query)
     contexts = [doc["content"] for doc in top_docs]
     ground_truth = "paid twice in a month"
     answer = generate_answer(query, contexts)


     print(" Retrieved Contexts:")
     for i, c in enumerate(contexts, start=1):
         print(f"{i}. {c}")

     print("\n LLM Answer:")
     print(answer)

     return {
         "query": query,
         "contexts": contexts,
         "ground_truth": ground_truth,
         "answer": answer
     }


# # ---------------------------
# # 5. Example Run
# # ---------------------------
if __name__ == "__main__":
    
    filename = "/Users/chandurajana/Documents/Chandu_Training/2025_09_GenAI_chandu/session6/student_chandu_chat_results.json"
    response = {}
    all_data = []

    while True:
    # Ask user for a question
        user_prompt = input("Ask something: ")
        if (user_prompt.lower() != 'quit'):
            query = user_prompt
            response = retrieve_and_generate_response(query)
            print(response)

            
            with open(filename, 'r') as file:
                all_data = json.load(file)
                #json.dump(response, file, indent=4)

            # Append new Q&A
               # Load existing data safely
            try:
                with open(filename, 'r') as f:
                    all_data = json.load(f)
            except FileNotFoundError:
                all_data = []
            except json.JSONDecodeError:
                # File exists but empty or not valid JSON
                all_data = []
        
            # Normalize to a list
            if isinstance(all_data, dict):
                # If the dict looks like {"entries": [...]} keep that list
                if 'entries' in all_data and isinstance(all_data['entries'], list):
                    all_data = all_data['entries']
                else:
                    # Wrap a single dict as the first list element
                    all_data = [all_data]
            elif not isinstance(all_data, list):
                all_data = []
            # Write updated data back

            all_data.append({"question": query, "answer": response})
            with open(filename, 'w') as file:
                json.dump(all_data, file, indent=4)
        else:
            break  


    #query = "How often do employees get paid?"
    #response = retrieve_and_generate_response(query)
    #print(response)
    #filename = "student_chandu_chat_results.json"
    #with open(filename, 'w') as file:
        #json.dump(response, file, indent=4)
