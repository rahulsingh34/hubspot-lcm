import dotenv
from transformers import LongformerTokenizer, LongformerModel
import torch
from langchain_pinecone import PineconeVectorStore
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_experimental.utilities import PythonREPL
from fastapi import FastAPI
from pydantic import BaseModel

# FastAPI initialization
app = FastAPI()

# Load environment variables
dotenv.load_dotenv()

# Load Longformer model and tokenizer
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-large-4096')
model = LongformerModel.from_pretrained('allenai/longformer-large-4096')

# Define custom embedding class with embed_query method
class LongformerEmbedding:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def embed_query(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=4096, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Mean pooling
        return embedding

# Instantiate the custom embedding class
longformer_embedder = LongformerEmbedding(tokenizer, model)

# Connect to Pinecone and initialize with the existing index
index_name = "hubspot-crm-txts"

# Initialize PineconeVectorStore with the custom embedding class
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=longformer_embedder  # Pass the embedding class here
)

# Set up the Retrieval Chain with LangChain
llm_rag = ChatAnthropic(model='claude-3-5-sonnet-20241022', temperature=0.1)

# Define your custom prompt with context and prefix
rag_template = """
Role: HubSpot code generator that prioritizes context-provided information.

Context: {text}

Generate Python code for HubSpot API operations that:
- Uses HubSpot Client Library when possible, falls back to requests library for associations
- Returns raw JSON responses (no parsing)
- Includes proper error handling
- Uses provided access_token
- Includes necessary imports
- Ends with print(response)

For filtering:
- Use HubSpot Filter objects with correct imports
- Match filter properties to object type

Code requirements:
- No comments/docstrings
- No syntax blocks
- No explanations
- Try-except wrapped
- Raw response only

For associations:
- Use requests.get() with proper endpoints
- Format: api.hubapi.com/crm/v4/objects/<object_type>/<id>/associations/<to_object_type>
- Include Authorization header with Bearer token

Question: {question}
"""

rag_prompt = PromptTemplate(template=rag_template, input_variables=["text", "question"])

# Define the Retrieval-Augmented Generation (RAG) chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm_rag,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs = {
        "prompt": rag_prompt,
        "document_variable_name": "text"
    },   
)

# Setup simple llm to explain the results
llm_explanation = ChatAnthropic(model='claude-3-5-haiku-20241022', temperature=0.5)

explanation_template = """
Explain HubSpot responses in plain language:

Format:
1. Summary line: "Found [X] results for [query type]"
2. Bullet-point details for each result
3. For errors: Explain issue and solution simply

Rules:
- Use everyday language
- No technical terms (API, JSON, etc.)
- Be thorough but simple
- No clarification offers
- No jargon

Style:
- Clear
- Descriptive
- Easy to understand
- Direct

Response: {response}

Use the response to answer the user's query below.

Query: {query}
"""

explanation_prompt = PromptTemplate(template=explanation_template, input_variables=["response", "query"])

# Define the explanation chain
explanation_chain = LLMChain(
    llm=llm_explanation,
    prompt=explanation_prompt
)

# Request model
class QueryRequest(BaseModel):
    question: str
    access_token: str

# API Endpoints
@app.post("/query")
async def query_model(request: QueryRequest):
    try:
        # Get access token
        access_token = request.access_token

        # Query RAG chain
        query = request.question
        rag_response = rag_chain.invoke(query + "| my access token is " + access_token)

        # Run explanation chain
        explanation = explanation_chain.invoke({
            "response": PythonREPL().run(rag_response['result']),
            "query": query
        })

        return {
            "success": True, 
            "query": query,
            "response": rag_response['result'],
            "explanation": explanation,
        }

    except Exception as e:
        return {
            "success": False, 
            "error": str(e)
        }