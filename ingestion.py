import asyncio
import logging as log
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

from logger import Colors


# Configure SSL context to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
vectorstore = PineconeVectorStore(index_name="documentation-helper", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap()
tavily_crawl = TavilyCrawl()


async def index_documents_async(docs: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously and add them to the vector store."""
    log.header("Indexing Documents Asynchronously")

    batches = [
        docs[i:i + batch_size] for i in range(0, len(docs), batch_size)
    ]
    async def add_batches(batches: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batches, async_mode=True)
        except Exception as e:
            log.error(f"Error adding batch {batch_num}: {e}", Colors.RED)
            return False
        return True
    tasks = [add_batches(batch, i) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        if result is False:
            log.error(f"Batch {i} failed to index.", Colors.RED)
        else:
            log.info(f"Batch {i} indexed successfully.", Colors.GREEN)


async def main():
    log.header("DOCUMENTATION HELPER - INGESTION")

    log.info(
        """TavilyCrawl: Starting to Crawl documents from the web...""",
        Colors.PURPLE,
    )

    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 1,
        "extract_depth": "advanced",
        "instructions": "content on ai agents"
    })

    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in res["results"]]

    log.info(f"TavilyCrawl: Finished crawling. Found {len(all_docs)} documents.", Colors.GREEN)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)

if __name__ == "__main__":
    asyncio.run(main())