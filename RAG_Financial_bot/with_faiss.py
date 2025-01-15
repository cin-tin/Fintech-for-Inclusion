import pandas as pd
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch.cuda.amp import autocast, GradScaler

# Initialize Tokenizer and Model
hf_token = "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"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
model = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if the model fits in GPU memory, otherwise use CPU
try:
    model.to(device)
except RuntimeError as e:
    print("CUDA out of memory, switching to CPU")
    device = torch.device("cpu")
    model.to(device)

# Load FAQ Dataset from CSV with Error Handling
def load_faq_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        documents = []
        for _, row in df.iterrows():
            documents.append(f"Q: {row['question']} A: {row['answer']}")
        return documents
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV: {e}")
        return []

# Assuming your CSV file is named 'Dataset.csv'
faq_documents = load_faq_from_csv("Dataset(copy).csv")

# Generate embeddings in batches with mixed precision if on GPU
def get_embeddings(documents, batch_size=8):
    embeddings = []
    scaler = GradScaler()
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU
        with torch.no_grad():
            with autocast():
                outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

embeddings = get_embeddings(faq_documents)

# Create Faiss index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Function to Query and Retrieve Top 5 Questions and Answers
def get_top_faq_responses_faiss(query):
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU
    with torch.no_grad():
        with autocast():
            query_embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, 5)
    
    formatted_responses = []
    for idx in indices[0]:
        text = faq_documents[idx]
        question = text.split("Q: ")[1].split(" A: ")[0]
        answer = text.split(" A: ")[1]
        formatted_responses.append({
            "question": question,
            "answer": answer
        })
    return formatted_responses

# Example Query
query = "What is tax evasion?"
responses = get_top_faq_responses_faiss(query)
for i, response in enumerate(responses, 1):
    print(f"Q{i}: {response['question']}\nA{i}: {response['answer']}\n")
