import importlib.metadata
packages = [
    "langchain",
    "langgraph",
    "tavily-python",
    "polygon",
    "langchain_community",
    "langchain_google_genai",
    "streamlit",
    "fastapi[all]",
    "uvicorn",
    "langchain-pinecone",
    "pypdf",
    "pyyaml",
    "requests",
    "yfinance",
    "tenacity",
    "scikit-learn",
    "langchain_groq",
    "langchain-tavily",
    "redis",
    "dotenv",
    'pydantic'
    ]
for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} (not installed)")