import pandas as pd
from transformers import AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
from llama_index.core.schema import Document

# Initialize Tokenizer and Model
hf_token = "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"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token)

# Only add the EOS token ID as stopping ID
stopping_ids = [tokenizer.eos_token_id]

llm = HuggingFaceLLM(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"token": hf_token, "torch_dtype": torch.bfloat16},
    generate_kwargs={"do_sample": True, "temperature": 0.6, "top_p": 0.9},
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer_kwargs={"token": hf_token},
    stopping_ids=stopping_ids,
)

# Load FAQ Dataset from CSV
def load_faq_from_csv(file_path):
    df = pd.read_csv(file_path)
    documents = []
    for _, row in df.iterrows():
        documents.append(Document(text=f"Q: {row['question']} A: {row['answer']}"))
    return documents

# Assuming your CSV file is named 'asset_management_(copy).csv'
faq_documents = load_faq_from_csv("Dataset(copy).csv")

# Set Embedding Model and LLM
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.llm = llm

# Create Index from Documents
index = VectorStoreIndex.from_documents(faq_documents)

# Create Query Engine with Top 5 Similarity Results
query_engine = index.as_query_engine(similarity_top_k=5)

# Function to Query and Retrieve Top 5 Questions and Answers
def get_top_faq_responses(query):
    response = query_engine.query(query)
    source_nodes = response.source_nodes[:5]  # Get top 5 results
    formatted_responses = []
    for node_with_score in source_nodes:
        text = node_with_score.node.text
        question = text.split("Q: ")[1].split(" A: ")[0]
        answer = text.split(" A: ")[1]
        formatted_responses.append({
            "question": question,
            "answer": answer
        })
    return formatted_responses

# Example Query
query = "what is tax evasion?"
responses = get_top_faq_responses(query)
for i, response in enumerate(responses, 1):
    print(f"Q{i}: {response['question']}\nA{i}: {response['answer']}\n")
