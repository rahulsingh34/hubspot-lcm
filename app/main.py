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
rag_template = """You are a helpful assistant that generates HubSpot API code based on the provided context.
    Always prioritize information from the context when available.

    Context: {text}

    Generate the Python code using the HubSpot Client Library with no comments to answer the following question.
    You only generate Python code no additional text, comments, docstrings, explanations, syntax code blocks.
    Use your general knowledge as a helpful assistant if no specific context is provided.
    Only return code, no additional text. Use the HubSpot Python library where possible.
    You will be provided with the access_token so be sure to use it. 
    Note that the returned response from your code should be a json object, do not parse it. 
    Your final line should be: print(response), where response is the json object returned from your API call.
    Do not add a 'limit' parameter within the response unless explicitly asked.
    If you are asked to filter the data by a specific property, you can create a Filter from the Hubspot Python library. 
    Don't forget to import the proper library for the Filter based on the HubSpot object in question and be sure to use the proper arguments for the API call.
    Be sure to wrap the code in a try catch block and print the error if any.

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
    You are a helpful assistant that explains JSON responses from the HubSpot API.
    Do not use technical terms in your response like API or JSON, explain in plain language. Be descriptive and easy to understand and thorough.
    Do not offer to provide clarifications at the end of your response.
    Your response should be formatted as such: an opening line that summarizes the number of results found and what the query was about, then a bullet point list with details about the results.
    If the response is an error, explain the error and how to fix it in the response.

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

# API Endpoints
@app.post("/query")
async def query_model(request: QueryRequest):
    try:
        # Query RAG chain
        query = request.question
        rag_response = rag_chain.invoke(query)

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