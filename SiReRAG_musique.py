import os
import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from raptor_pack.llama_index.packs.raptor.base import RaptorRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import json
from datasets import load_dataset
from llama_index.core.schema import Document


from llama_index.llms.deepseek import DeepSeek

llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DS_API"))


from llama_index.embeddings.ollama import OllamaEmbedding
ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

# 添加一个简单的测试
print("Testing Ollama embedding...")
try:
    test_text = "This is a test."
    embedding = ollama_embedding.get_text_embedding(test_text)
    print("Ollama embedding test successful!")
except Exception as e:
    print(f"Ollama embedding test failed: {str(e)}")


os.environ["OPENAI_API_KEY"] = ""
nest_asyncio.apply()
with open('musique_corpus.json') as file:
	data = file.read()
	lines = json.loads(data)


output_directory = 'MuSiQue_temp_data'
os.makedirs(output_directory, exist_ok=True)

all_file, count = [], 0
for value in lines:
	file_path = os.path.join(output_directory, f"{count}.txt")
	with open(file_path, 'w') as file:
		file_str = value['title'] + '\n' + value['text']
		file.write(file_str)
	all_file.append(file_path)
	count += 1

corpus = json.load(open("musique_kg.json"))
documents = []
entities_facts = {}
fact_counts = {}
for doc in corpus:
    for rel in doc["facts"]:
        documents.append(Document(text=rel["fact"]))
        for ent in rel["entities"]:
            if ent.lower() not in entities_facts:
                entities_facts[ent.lower()] = []
            entities_facts[ent.lower()] += [rel["fact"]]
            if rel["fact"] not in fact_counts:
                fact_counts[rel["fact"]] = 0
            fact_counts[rel["fact"]] += 1
new_docs = set()
for ent in entities_facts:
    if len(entities_facts[ent]) == 1 and fact_counts[entities_facts[ent][0]] > 1:
        continue
    new_docs.add("\n".join(entities_facts[ent]))
higher_level_facts = []
for doc in new_docs:
     higher_level_facts.append(Document(text=doc))

documents = SimpleDirectoryReader(input_files=all_file).load_data()
retriever = RaptorRetriever(documents, higher_level_facts=higher_level_facts, embed_model=ollama_embedding, llm=llm, similarity_top_k=20, mode="collapsed", verbose = True)
query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)