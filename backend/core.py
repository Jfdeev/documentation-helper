from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = "langchain-docs"

def run_llm(query: str):
    embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
    docs_search = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    llm = ChatOllama(model="gemma3:1b")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt=retrieval_qa_chat_prompt)

    # Create the retrieval
    qa = create_retrieval_chain(
        retriever=docs_search.as_retriever(),
        combine_docs_chain=stuff_documents_chain,
    )
    result = qa.invoke(input={"input": query})
    return result

if __name__ == "__main__":
    query = "What is LangChain?"
    result = run_llm(query)
    print(result['answer'])