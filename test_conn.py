from langchain_google_genai import GoogleGenerativeAIEmbeddings
emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key="AIzaSyBzotrB7B4c1I2N_7IuvG4Bnl7yCl4hbMw")
print(emb.embed_query("hello world"))
