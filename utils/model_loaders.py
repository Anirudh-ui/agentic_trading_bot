import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from vertexai.language_models import TextEmbeddingModel
from typing import Dict, Any, List 
from langchain_core.embeddings import Embeddings

class VertexAIEmbeddings_for_workflow(Embeddings):
    """Adapter to make Vertex `text-embedding-004` compatible with LangChain."""

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results = self.model.get_embeddings(texts)
        return [emb.values for emb in results]

    def embed_query(self, text: str) -> List[float]:
        result = self.model.get_embeddings([text])[0]
        return result.values
class ModelLoader:
    """
    A utility class to load embedding models and LLM models.
    """
    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config=load_config()

    def _validate_env(self):
        """
        Validate necessary environment variables.
        """
        required_vars = ["GOOGLE_API_KEY","GROQ_API_KEY"]
        self.groq_api_key=os.getenv("GROQ_API_KEY")
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

    def load_embeddings(self):
        """
        Load and return the embedding model.
        """
        print("Loading Embedding model")
        model_name=self.config["embedding_model"]["model_name"]
        return GoogleGenerativeAIEmbeddings(model=model_name)

    def load_llm(self):
        """
        Load and return the LLM model.
        """
        print("LLM loading...")
        model_name=self.config["llm"]["groq"]["model_name"]
        print("******this is my key*****")
        print(self.groq_api_key)
        groq_model=ChatGroq(model=model_name,api_key=self.groq_api_key)
        
        
        return groq_model  # Placeholder for future LLM loading
    


    def load_embeddings_vertex(self):
        """
        Load and return the Vertex AI embedding model (text-embedding-004).
        """
        print("Loading Embedding model (Vertex text-embedding-004)")
        
        # Load the raw Vertex AI model
        base_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        
        # Wrap it so LangChain can use it
        return VertexAIEmbeddings_for_workflow(base_model)
