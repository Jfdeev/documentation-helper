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
from langchain_ollama import OllamaEmbeddings
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


def index_documents_sync(docs: List[Document], batch_size: int = 10):
    """Process documents in batches synchronously and add them to the vector store."""
    print(f"Indexing {len(docs)} documents in batches of {batch_size}...")

    batches = [
        docs[i:i + batch_size] for i in range(0, len(docs), batch_size)
    ]
    
    successful = 0
    for i, batch in enumerate(batches):
        try:
            print(f"Processing batch {i+1}/{len(batches)}...")
            vectorstore.add_documents(batch)
            successful += 1
            print(f"✓ Batch {i+1} completed successfully")
        except Exception as e:
            print(f"✗ Error adding batch {i+1}: {e}")

    if successful == len(batches):
        print(f"✓ Successfully indexed all {len(docs)} documents in {len(batches)} batches.")
    else:
        print(f"⚠ Indexed {successful} out of {len(batches)} batches successfully.")
    
    return successful == len(batches) 


async def main():
    log.info("DOCUMENTATION HELPER - INGESTION")

    log.info(
        """TavilyCrawl: Starting to Crawl documents from the web...""",
        Colors.PURPLE,
    )

    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/docs/introduction/",
        "max_depth": 1,
        "include_domains": ["python.langchain.com"],
        "extract_depth": "basic",
        "instructions": "Extract documentation content about LangChain framework, AI agents, and tutorials"
    })

    # Debug completo
    print("DEBUG - Estrutura do resultado:")
    print(f"Chaves disponíveis: {res.keys()}")
    print(f"Total de resultados: {len(res.get('results', []))}")
    
    if res.get("results"):
        for i, result in enumerate(res["results"][:3]):  # Mostrar apenas os primeiros 3
            print(f"\n--- Resultado {i+1} ---")
            print(f"URL: {result.get('url', 'N/A')}")
            print(f"Chaves disponíveis no resultado: {list(result.keys())}")
            
            # Verificar todos os campos possíveis
            raw_content = result.get("raw_content", "")
            content = result.get("content", "")
            text = result.get("text", "")
            
            print(f"raw_content length: {len(raw_content)}")
            print(f"content length: {len(content)}")
            print(f"text length: {len(text)}")
            
            if raw_content and len(raw_content) > 10:
                print(f"raw_content preview: {raw_content[:300]}...")
            if content and len(content) > 10:
                print(f"content preview: {content[:300]}...")
            if text and len(text) > 10:
                print(f"text preview: {text[:300]}...")

    # Tentar extrair conteúdo usando diferentes estratégias
    all_docs = []
    for result in res["results"]:
        content = ""
        url = result.get("url", "unknown")
        
        # Estratégia 1: raw_content
        if result.get("raw_content") and len(result["raw_content"].strip()) > 10 and result["raw_content"].strip() != "```":
            content = result["raw_content"]
            print(f"✓ Usando raw_content para {url}")
        # Estratégia 2: content
        elif result.get("content") and len(result["content"].strip()) > 10:
            content = result["content"]
            print(f"✓ Usando content para {url}")
        # Estratégia 3: text
        elif result.get("text") and len(result["text"].strip()) > 10:
            content = result["text"]
            print(f"✓ Usando text para {url}")
        else:
            print(f"❌ Nenhum conteúdo válido encontrado para {url}")
            continue
        
        # Limpar conteúdo
        content = content.strip()
        if content and len(content) > 50:  # Apenas conteúdo substancial
            all_docs.append(Document(
                page_content=content,
                metadata={"source": url}
            ))
            print(f"✅ Adicionado documento com {len(content)} caracteres")

    print(f"\n📊 RESUMO:")
    print(f"Total de URLs crawled: {len(res.get('results', []))}")
    print(f"Documentos com conteúdo válido: {len(all_docs)}")

    if len(all_docs) == 0:
        print("❌ ERRO: Nenhum documento válido foi extraído!")
        print("Vamos tentar uma URL mais específica...")
        return

    log.info(f"TavilyCrawl: Finished crawling. Found {len(all_docs)} documents with valid content.", Colors.GREEN)
    
    # Filtrar documentos antes de dividir
    print("Filtrando conteúdo antes da divisão...")
    filtered_docs = []
    for doc in all_docs:
        # Limpar conteúdo de código blocks vazios
        content = doc.page_content
        
        # Remover blocos de código vazios
        lines = content.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and line != "```" and not line.startswith("```\n"):
                clean_lines.append(line)
        
        clean_content = '\n'.join(clean_lines)
        
        if len(clean_content.strip()) > 100:  # Apenas conteúdo substancial
            filtered_docs.append(Document(
                page_content=clean_content,
                metadata=doc.metadata
            ))
            print(f" Documento filtrado: {len(clean_content)} caracteres")
        else:
            print(f" Documento rejeitado por ser muito pequeno: {len(clean_content)} caracteres")
    
    print(f"📊 Documentos após filtragem: {len(filtered_docs)}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(filtered_docs)
    
    # Filtrar chunks pequenos ou vazios
    final_docs = []
    for doc in split_docs:
        content = doc.page_content.strip()
        if len(content) > 50 and content != "```" and not content.startswith("```"):
            final_docs.append(doc)
    
    print(f"📊 Documentos finais após divisão e filtragem: {len(final_docs)}")

    index_documents_sync(final_docs, batch_size=10)

    log.info("Ingestion Complete")
    log.info(f"Total documents indexed: {len(split_docs)}", Colors.GREEN)
    log.info(f"Total documents crawled: {len(all_docs)}", Colors.GREEN)

if __name__ == "__main__":
    asyncio.run(main())