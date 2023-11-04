import weaviate
import os
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_KEY = os.getenv("WEAVIATE_KEY")  
auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_KEY)
client = weaviate.Client(url="https://diagnosis-ai-generator-kk6m9y1x.weaviate.network", auth_client_secret=auth_config)  

client.schema.get()