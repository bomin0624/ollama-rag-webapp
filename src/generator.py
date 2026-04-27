import os
from functools import lru_cache

import ollama
from config import embedding_model, reranker_model
from retriever import HybridRetriever, RAGRetriever, initialize_vector_database

DB_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "vectordatabase")

# Using LRU cache to store the retriever instance for efficient reuse across multiple calls
@lru_cache(maxsize=1)
def get_retriever(db_directory: str) -> RAGRetriever:
    """Gets or creates a cached RAGRetriever instance."""
    return RAGRetriever(
            db_directory=db_directory,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )


def generate_prompt_stream(query:str) -> str:
    prompt = f"\nBased on the following query: {query} and the context provided below to give the user answer.\n"
    # Ensure the database is initialized (idempotent check)
    initialize_vector_database(DB_DIRECTORY)
    retriever = get_retriever(DB_DIRECTORY)
    retrieved_docs = retriever.retrieve_and_rerank(query)

    if not retrieved_docs:
        prompt += "\nNo relevant documents found.\n"
        return prompt
    else:
        for idx, doc in enumerate(retrieved_docs):
            # print(f"\n--- Document {idx + 1} ---")
            # print(f"Content: {doc.page_content[:250]}...")
            # print(f"Metadata: {doc.metadata}")
            prompt += f"\nDocument {doc.metadata['id']}:\n{doc.page_content}\n"
    return prompt

def generate_response(query: str) -> str:
    prompt = generate_prompt_stream(query)
    result = ollama.chat(
        model='llama3.1',
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    return result['message']['content']

# def test_ollama():
#     """測試 Ollama 連接和功能"""
    
#     print("=" * 60)
#     print("🔍 測試 Ollama 連接")
#     print("=" * 60)
    
#     try:
#         # 1. 測試列出模型
#         print("\n1️⃣ 檢查可用模型...")
#         response = ollama.list()
#         print(f"原始響應: {response}\n")
        
#         if 'models' in response:
#             print("✅ 找到以下模型:")
#             for model in response['models']:
#                 # 安全獲取模型信息
#                 name = model.get('name', model.get('model', 'unknown'))
#                 size = model.get('size', 0) / (1024**3)  # 轉 GB
#                 modified = model.get('modified_at', 'unknown')
#                 print(f"   📦 {name}")
#                 print(f"      大小: {size:.2f} GB")
#                 print(f"      修改時間: {modified}")
#         else:
#             print("⚠️ 響應格式異常")
#             return
        
#         # 2. 測試基本生成
#         print("\n2️⃣ 測試基本生成...")
#         result = ollama.generate(
#             model='llama3.1',
#             prompt='Say "Hello" in one word'
#         )
#         print(f"✅ 生成結果: {result['response']}")
        
#         # 3. 測試對話模式
#         print("\n3️⃣ 測試對話模式...")
#         result = ollama.chat(
#             model='llama3.1',
#             messages=[
#                 {'role': 'user', 'content': '用繁體中文說你好，只回答一句話'}
#             ]
#         )
#         print(f"✅ 對話結果: {result['message']['content']}")
        
#         # 4. 測試 embeddings
#         print("\n4️⃣ 測試 Embeddings...")
#         result = ollama.embeddings(
#             model='llama3.1',
#             prompt='測試文本'
#         )
#         embedding = result['embedding']
#         print(f"✅ Embedding 維度: {len(embedding)}")
#         print(f"   前 5 個值: {embedding[:5]}")
        
#         # 5. 測試串流
#         print("\n5️⃣ 測試串流生成...")
#         print("回應: ", end="", flush=True)
#         stream = ollama.chat(
#             model='llama3.1',
#             messages=[{'role': 'user', 'content': '數到 5，用繁體中文'}],
#             stream=True
#         )
#         for chunk in stream:
#             content = chunk['message']['content']
#             print(content, end="", flush=True)
#         print()
        
#         print("\n" + "=" * 60)
#         print("🎉 所有測試通過！")
#         print("=" * 60)
        
#     except Exception as e:
#         print(f"\n❌ 錯誤: {e}")
#         import traceback
#         traceback.print_exc()


if __name__ == "__main__":
    # test_ollama()
    query = input("Please enter your query: ")
    response = generate_response(query)
    print(response)