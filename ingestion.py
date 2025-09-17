import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_tavily import TavilyClient, TavilyExtract, TavilyMap, TavilyCrawl


# Configure SSL context to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
vectorstore = PineconeVectorStore(index_name="documentation-helper", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap()
tavily_crawl = TavilyCrawl()


async def main():
    print("Hello, World!")



if __name__ == "__main__":
    asyncio.run(main())