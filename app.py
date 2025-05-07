import os
import uuid
import torch
import fitz  # PyMuPDF
import pandas as pd
import textwrap
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize

# 下載NLTK資源(如果使用句子分割)
nltk.download('punkt', quiet=True)

# 載入 .env 檔案中的環境變數
load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#################################################
# 可調整參數區域 - 修改這些值來測試不同設定
#################################################

# [文本處理參數]
CHUNK_SIZE = 500       # 文本片段大小 (字符數): 100-300(精確度高), 400-700(平衡), 800-1500(上下文多)
OVERLAP_SIZE = 0       # 文本片段重疊字符數: 0(無重疊), 50-100(小重疊), 100-250(大重疊)
CHUNK_STRATEGY = "character"  # 分割策略: "character"(按字符) 或 "sentence"(按句子)

# [檢索參數]
TOP_K_RESULTS = 3      # 檢索結果數量: 1-2(聚焦), 3-5(平衡), 6-10(廣泛)
USE_RERANKER = False   # 是否使用再排序: True(使用), False(不使用)

# [模型選擇參數]
# 嵌入模型選項
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 高質量通用型
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 更快速較小
# EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"  # 問答優化
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 多語言

# 問答/生成模型選項 (注意用太大會下載很久喔)
QA_MODEL = "google/flan-t5-base"  # 平衡速度與質量
# QA_MODEL = "google/flan-t5-small"  # 更快但能力有限
# QA_MODEL = "google/flan-t5-large"  # 更高質量但需更多資源
# QA_MODEL = "google/flan-t5-xl"  # 高質量，需要強大GPU

# [生成參數]
PROMPT_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"  # 提示詞模板
MAX_NEW_TOKENS = 100   # 生成回答的最大長度: 50(簡短), 100(中等), 200(詳細)

#################################################
# 程式主體 - 一般情況下不需要修改
#################################################

# 建立 Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_docs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 載入模型
print(f"正在載入嵌入模型: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print(f"正在載入問答模型: {QA_MODEL}")
qa_model = pipeline(
    "text2text-generation",
    model=QA_MODEL,
    tokenizer=QA_MODEL,
    device=0 if torch.cuda.is_available() else -1
)

# 載入再排序器(如果啟用)
reranker = None
if USE_RERANKER:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("已載入再排序器")

# 儲存文件段落與對應嵌入
doc_chunks = []
doc_embeddings = []

def chunk_text(text, strategy="character"):
    """根據選擇的策略將文本分割成片段"""
    if strategy == "sentence":
        # 基於句子的分割
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len <= CHUNK_SIZE:
                current_chunk.append(sentence)
                current_length += sentence_len
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_len
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        # 處理重疊
        if OVERLAP_SIZE > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i == 0:
                    overlapped_chunks.append(chunks[i])
                else:
                    # 嘗試從上一個塊獲取重疊的文本
                    prev_chunk = chunks[i-1]
                    overlap_text = ""
                    if len(prev_chunk) >= OVERLAP_SIZE:
                        overlap_text = prev_chunk[-OVERLAP_SIZE:]
                    overlapped_chunks.append(overlap_text + chunks[i])
            return overlapped_chunks
        
        return chunks
    else:
        # 默認: 基於字符的分割
        if OVERLAP_SIZE > 0:
            chunks = []
            for i in range(0, len(text), CHUNK_SIZE - OVERLAP_SIZE):
                chunks.append(text[i:i + CHUNK_SIZE])
            return chunks
        else:
            return [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

# 文件文字抽取
def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        doc = fitz.open(filepath)
        return "\n".join([page.get_text() for page in doc])
    elif ext == ".csv":
        df = pd.read_csv(filepath)
        return df.to_string(index=False)
    else:
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/settings", methods=["GET"])
def get_settings():
    """返回當前系統設定"""
    return jsonify({
        "chunk_size": CHUNK_SIZE,
        "overlap_size": OVERLAP_SIZE,
        "chunk_strategy": CHUNK_STRATEGY,
        "top_k_results": TOP_K_RESULTS,
        "use_reranker": USE_RERANKER,
        "embedding_model": EMBEDDING_MODEL,
        "qa_model": QA_MODEL,
        "max_new_tokens": MAX_NEW_TOKENS,
        "total_chunks": len(doc_chunks)
    })

@app.route("/settings", methods=["POST"])
def update_settings():
    """更新RAG系統設定"""
    global CHUNK_SIZE, OVERLAP_SIZE, CHUNK_STRATEGY, TOP_K_RESULTS, USE_RERANKER, MAX_NEW_TOKENS
    
    data = request.get_json()
    settings_changed = False
    
    if "chunk_size" in data:
        CHUNK_SIZE = int(data["chunk_size"])
        settings_changed = True
    
    if "overlap_size" in data:
        OVERLAP_SIZE = int(data["overlap_size"])
        settings_changed = True
    
    if "chunk_strategy" in data:
        if data["chunk_strategy"] in ["character", "sentence"]:
            CHUNK_STRATEGY = data["chunk_strategy"]
            settings_changed = True
    
    if "top_k_results" in data:
        TOP_K_RESULTS = int(data["top_k_results"])
        settings_changed = True
    
    if "use_reranker" in data:
        new_reranker_setting = bool(data["use_reranker"])
        if new_reranker_setting != USE_RERANKER:
            USE_RERANKER = new_reranker_setting
            if USE_RERANKER and reranker is None:
                from sentence_transformers import CrossEncoder
                reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            settings_changed = True
    
    if "max_new_tokens" in data:
        MAX_NEW_TOKENS = int(data["max_new_tokens"])
        settings_changed = True
        
    return jsonify({
        "message": "設定已更新" if settings_changed else "沒有設定被改變",
        "current_settings": {
            "chunk_size": CHUNK_SIZE,
            "overlap_size": OVERLAP_SIZE,
            "chunk_strategy": CHUNK_STRATEGY,
            "top_k_results": TOP_K_RESULTS,
            "use_reranker": USE_RERANKER,
            "max_new_tokens": MAX_NEW_TOKENS
        }
    })

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "沒有上傳檔案"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".txt", ".pdf", ".csv"]:
        return jsonify({"error": "不支援的檔案格式"}), 400

    filename = str(uuid.uuid4()) + ext
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    content = extract_text(filepath)
    if not content:
        return jsonify({"error": "無法從檔案中提取文字"}), 500

    # 使用指定策略分割文本
    chunks = chunk_text(content, CHUNK_STRATEGY)
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)

    doc_chunks.extend(chunks)
    doc_embeddings.extend(embeddings)

    return jsonify({
        "message": f"{file.filename} 已成功上傳並處理。",
        "stats": {
            "chunks_created": len(chunks),
            "chunk_size": CHUNK_SIZE,
            "overlap_size": OVERLAP_SIZE,
            "chunk_strategy": CHUNK_STRATEGY,
            "total_chunks": len(doc_chunks)
        }
    })

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("query")
    if not question:
        return jsonify({"error": "沒有提供查詢內容"}), 400

    if not doc_embeddings:
        return jsonify({"error": "尚未上傳任何文件。"}), 400

    # 將問題轉換為向量
    query_embedding = embedding_model.encode(question, convert_to_tensor=True)
    
    # 使用語義搜索找出相關文本
    hits = util.semantic_search(query_embedding, doc_embeddings, top_k=TOP_K_RESULTS)[0]
    
    # 如果啟用再排序
    if USE_RERANKER and reranker:
        # 準備再排序的候選項
        candidate_pairs = [(question, doc_chunks[hit["corpus_id"]]) for hit in hits]
        rerank_scores = reranker.predict(candidate_pairs)
        
        # 重新排序候選項
        reranked_results = [(hits[i]["corpus_id"], float(rerank_scores[i])) for i in range(len(hits))]
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # 使用再排序結果
        retrieved_chunks = [
            {
                "chunk_id": chunk_id,
                "similarity": score,
                "preview": textwrap.shorten(doc_chunks[chunk_id], width=100, placeholder="...")
            }
            for chunk_id, score in reranked_results
        ]
        
        context = "\n".join([doc_chunks[chunk_id] for chunk_id, _ in reranked_results])
    else:
        # 使用原始排序結果
        retrieved_chunks = [
            {
                "chunk_id": hit["corpus_id"],
                "similarity": float(hit["score"]),
                "preview": textwrap.shorten(doc_chunks[hit["corpus_id"]], width=100, placeholder="...")
            }
            for hit in hits
        ]
        
        context = "\n".join([doc_chunks[hit["corpus_id"]] for hit in hits])

    # 使用提示詞模板生成回答
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    result = qa_model(prompt, max_new_tokens=MAX_NEW_TOKENS)[0]["generated_text"]

    return jsonify({
        "response": result,
        "debug_info": {
            "chunk_size": CHUNK_SIZE,
            "overlap_size": OVERLAP_SIZE,
            "chunk_strategy": CHUNK_STRATEGY,
            "top_k": TOP_K_RESULTS,
            "use_reranker": USE_RERANKER,
            "retrieved_chunks": retrieved_chunks,
            "context_length": len(context)
        }
    })

@app.route("/clear", methods=["POST"])
def clear_data():
    """清除已上傳的文檔資料"""
    global doc_chunks, doc_embeddings
    doc_chunks = []
    doc_embeddings = []
    return jsonify({"message": "所有文檔資料已清除"})

if __name__ == "__main__":
    print(f"RAG系統已啟動，使用以下設定:")
    print(f"- 文本片段大小: {CHUNK_SIZE} 字符")
    print(f"- 文本片段重疊: {OVERLAP_SIZE} 字符")
    print(f"- 分割策略: {CHUNK_STRATEGY}")
    print(f"- 檢索結果數量: {TOP_K_RESULTS}")
    print(f"- 使用再排序: {USE_RERANKER}")
    print(f"- 嵌入模型: {EMBEDDING_MODEL}")
    print(f"- 問答模型: {QA_MODEL}")
    app.run(debug=True)