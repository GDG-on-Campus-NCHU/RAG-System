您說得對，安裝說明需要更加明確。讓我修正這部分：

# RAG-System

一個基於檢索增強生成(RAG)的問答系統，能從上傳的文檔中提取相關知識回答用戶問題。使用Python、Flask和Hugging Face模型構建。

## 功能特點

- **文檔上傳**：支援PDF、TXT和CSV文件格式
- **語義搜索**：使用向量嵌入找出最相關的信息
- **可自定義參數**：調整文本片段大小、重疊量、檢索數量等
- **互動界面**：簡潔的網頁界面用於文檔上傳和問答
- **調試信息**：查看相似度分數和檢索到的上下文以增加透明度

## 開始使用

### 環境要求

- Python 3.7+
- pip套件管理器

### 安裝步驟

1. 選擇一個您喜歡的位置，創建專案資料夾：
   ```bash
   # 建立專案資料夾
   mkdir rag_test
   cd rag_test
   ```

2. 克隆存儲庫到此資料夾：
   ```bash
   git clone https://github.com/GDG-on-Campus-NCHU/RAG-System.git .
   ```

3. 在專案資料夾外創建虛擬環境資料夾：
   ```bash
   # 回到上一層目錄
   cd ..
   # 創建虛擬環境資料夾
   python -m venv rag_env
   ```

4. 啟動虛擬環境：
   ```bash
   # Windows系統
   rag_env\Scripts\activate
   # macOS/Linux系統
   source rag_env/bin/activate
   ```

5. 進入專案資料夾並安裝依賴：
   ```bash
   cd rag_test
   pip install -r requirements.txt
   ```

6. 在項目根目錄創建`.env`文件，添加您的Hugging Face API令牌：
   ```
   HUGGINGFACEHUB_API_TOKEN=您的令牌
   ```

### 運行應用

啟動伺服器：
```bash
python app.py
```

訪問網頁界面：http://localhost:5000

## 使用方法

1. **上傳文檔**：使用「上傳知識庫」部分上傳您的文檔（PDF、TXT或CSV）。

2. **提問問題**：在「聊天界面」部分輸入您的問題，然後點擊「發送問題」。

3. **查看結果**：獲取基於您上傳文檔內容生成的回答。

## 自定義設置

您可以在`app.py`文件頂部調整各種參數：

- `CHUNK_SIZE`：文本片段大小（預設：500字符）
- `TOP_K_RESULTS`：檢索相關片段的數量（預設：3）
- `EMBEDDING_MODEL`：用於文本向量化的模型
- `QA_MODEL`：用於生成回答的模型

## 工作原理

1. 文檔被分割成較小的片段
2. 每個片段被轉換為向量嵌入
3. 當提出問題時，問題也被轉換為向量
4. 系統找出最相似的文檔片段
5. 問題和相關片段一起發送到語言模型
6. 模型基於提供的上下文生成回答

## 授權

本項目是開源的，基於MIT許可證發布。


