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
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_tavily import TavilyExtract, TavilyMap, TavilyCrawl

from logger import Colors

load_dotenv()

# Configure SSL context to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
vectorstore = PineconeVectorStore(index_name="langchain-docs", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap()
tavily_crawl = TavilyCrawl()


async def index_documents_async(docs: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously and add them to the vector store."""
    log.info(f"Indexing {len(docs)} documents in batches of {batch_size}...", Colors.PURPLE)

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

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log.info(f"Successfully indexed all {len(docs)} documents in {len(batches)} batches.", Colors.GREEN)
    else:
        log.warning(f"Indexed {successful} out of {len(batches)} batches successfully.", Colors.YELLOW) 


async def main():
    log.info("DOCUMENTATION HELPER - INGESTION")

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

    await index_documents_async(split_docs, batch_size=500)

    log.info("Ingestion Complete")
    log.info(f"Total documents indexed: {len(split_docs)}", Colors.GREEN)
    log.info(f"Total documents crawled: {len(all_docs)}", Colors.GREEN)

if __name__ == "__main__":
    asyncio.run(main())