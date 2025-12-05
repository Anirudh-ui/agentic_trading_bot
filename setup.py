from setuptools import find_packages,setup

setup(
    name="agentic-trading-system",
    version="0.0.1",
    author="Anirudh PVS",
    author_email="anirudhpvs98@gmail.com",
    packages=find_packages(),
    install_requires=[
        # Core Orchestration and Generative AI
        'langchain',
        'langgraph',
        'langchain_google_genai',
        'google-generativeai',
        'langchain_groq',
        
        # Data and Tooling
        'lancedb',
        'tavily-python',
        'polygon',
        'yfinance>=0.2.40',
        
        # Web/API
        'fastapi[all]',
        'uvicorn',
        'requests',
        
        # RAG and Data Processing
        'pypdf',
        'pdf2image',
        'pdftotext', # This will compile now that headers are installed
        'faiss-cpu',
        'spacy',
        'python-dateutil',
        'pydantic'
        
        # State/Memory
        'redis',
        'langgraph[redis]',
        'dotenv',
        'weaviate',
        'weaviate-client',
        'pinecone'
    ]
)