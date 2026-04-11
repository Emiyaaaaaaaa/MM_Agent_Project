import os
import json
import time
from pathlib import Path
from datetime import datetime
from collections import Counter
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ================= 配置区 =================
client_ai = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
PROJECT_ROOT = Path(r"c:\Users\a2231\Desktop\RAG")
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "multimodal_papers"
QUERY_MODEL = "gemini-embedding-2-preview"

# 初始化 ChromaDB
client_db = chromadb.PersistentClient(path=str(DB_PATH))
collection = client_db.get_or_create_collection(name=COLLECTION_NAME)

# ================= 扩展测试集 (V2) =================
TEST_CASES = [
    {
        "id": "TC-001",
        "category": "2021 MCM A (Fungi)",
        "query": "What is the relationship between fungi decomposition rates and climate variants like temperature or moisture?",
        "expected": ["decomposition", "climate", "temperature", "fungi"]
    },
    {
        "id": "TC-002",
        "category": "2022 MCM C (Wordle)",
        "query": "How did teams use the ARIMA model to predict the number of reported Wordle results?",
        "expected": ["ARIMA", "predict", "Wordle", "time series"]
    },
    {
        "id": "TC-003",
        "category": "2023 MCM A (Plant-Pollinator)",
        "query": "Discuss the impact of environmental changes on plant-pollinator networks and extinction risk.",
        "expected": ["pollinator", "extinction", "network", "stability"]
    },
    {
        "id": "TC-004",
        "category": "2024 MCM F (Baltimore)",
        "query": "What are the key indicators for evaluating the fairness of public transportation in Baltimore?",
        "expected": ["Baltimore", "fairness", "transportation", "accessibility"]
    },
    {
        "id": "TC-005",
        "category": "2025 MCM F (Cybersecurity)",
        "query": "Which type of cybersecurity policy (e.g., D-N, I-C) is most effective in the first year of implementation?",
        "expected": ["D-N", "Domestic-Non-Collaborative", "effectiveness", "first year"]
    },
    {
        "id": "TC-006",
        "category": "Commentary Analysis",
        "query": "What are the common strengths of Outstanding Papers mentioned in the COMAP judges' commentary?",
        "expected": ["judges", "commentary", "Outstanding", "strengths", "model"]
    },
    {
        "id": "TC-007",
        "category": "Methodology / Algorithms",
        "query": "Show me the definition or application of the Gini coefficient in tournament fairness modeling.",
        "expected": ["Gini", "fairness", "inequality", "coefficient"]
    },
    {
        "id": "TC-008",
        "category": "Visual / Flowchart Search",
        "query": "Are there any segments that describe a flowchart or a model framework diagram?",
        "expected": ["Figure", "flowchart", "framework", "diagram"]
    },
    {
        "id": "TC-009",
        "category": "Data Pre-processing",
        "query": "Explain how teams handled missing data or data normalization in their modeling process.",
        "expected": ["missing", "normalize", "preprocessing", "interpolation"]
    },
    {
        "id": "TC-010",
        "category": "Sustainability / Ecology",
        "query": "How do models address the long-term sustainability of the Great Barrier Reef ecosystem?",
        "expected": ["Great Barrier Reef", "sustainability", "ecosystem", "coral"]
    }
]

def run_comprehensive_v2():
    print(f"{'='*80}")
    print(f" RAG 综合验证模块 V2 - 全量数据体检 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    total_docs = collection.count()
    print(f"[数据概况] 当前索引文档总数: {total_docs}")
    
    report = []
    year_stats = Counter()
    
    for tc in TEST_CASES:
        print(f"\n>>> [测试 {tc['id']}] {tc['category']}")
        print(f"    Q: {tc['query']}")
        
        start_t = time.time()
        try:
            # 1. 生成向量
            resp = client_ai.models.embed_content(
                model=QUERY_MODEL,
                contents=tc['query'],
                config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')
            )
            v = resp.embeddings[0].values
            
            # 2. 检索并获取 Top 3
            res = collection.query(query_embeddings=[v], n_results=3)
            latency = (time.time() - start_t) * 1000
            
            # 3. 分析结果
            best_dist = res['distances'][0][0]
            best_meta = res['metadatas'][0][0]
            best_doc = res['documents'][0][0]
            
            # 记录年份统计（命中分布）
            hit_years = [m.get('year') for m in res['metadatas'][0] if m.get('year')]
            year_stats.update(hit_years)
            
            # 关键词命中检查（在前 3 条中搜寻）
            all_text = " ".join(res['documents'][0])
            hit_keywords = [w for w in tc['expected'] if w.lower() in all_text.lower()]
            
            # 图片检查
            has_img = any(m.get("image_paths_json") for m in res['metadatas'][0])
            
            status = "PASS" if best_dist < 0.6 or hit_keywords else "REVIEW"
            
            print(f"    - 最佳匹配: {best_meta.get('filename')} (Dist: {best_dist:.4f})")
            print(f"    - 耗时: {latency:.1f}ms | 命中关键词: {len(hit_keywords)}/{len(tc['expected'])}")
            print(f"    - 关联图表: {'YES' if has_img else 'NO'}")
            
            report.append({
                "id": tc['id'], 
                "cat": tc['category'], 
                "dist": best_dist, 
                "status": status,
                "latency": latency,
                "keywords": f"{len(hit_keywords)}/{len(tc['expected'])}",
                "filename": best_meta.get('filename')
            })
            
        except Exception as e:
            print(f"    [!] 运行出错: {e}")
            report.append({"id": tc['id'], "cat": tc['category'], "status": "FAIL", "dist": 0, "latency": 0, "keywords": "0/0", "filename": "N/A"})

    # ================= 打印报告汇总表 =================
    print(f"\n\n{'='*30} 验证汇总报表 (Summary Report) {'='*30}")
    print(f"{'ID':<8} {'类别/年份':<25} {'状态':<10} {'相似度':<8} {'关键词':<8} {'来源文件'}")
    print("-" * 90)
    for r in report:
        print(f"{r['id']:<8} {r['cat']:<25} {r['status']:<10} {r['dist']:<8.4f} {r['keywords']:<8} {r['filename']}")
    
    print("-" * 90)
    print(f"[命中热力图] 检索覆盖年份分布: {dict(year_stats)}")
    success_rate = (sum(1 for r in report if r['status'] != "FAIL") / len(TEST_CASES)) * 100
    print(f"全链路技术通达率: {success_rate:.1f}%")
    print(f"{'='*90}")

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("错误：未找到 GOOGLE_API_KEY")
    else:
        run_comprehensive_v2()
