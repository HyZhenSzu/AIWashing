"""
scoring_chunks_iter.py

使用 ReportLoader.iter_files() 按文件迭代读取年报并对包含 AI 关键词的 chunk 打分。
输出 CSV 包含列：fname, firm_id, year, chunk_id, chunk_len, score, ai_flag

兼容断点续跑：脚本会读取输出 CSV 的最后一条有效数据行并从该位置继续。
"""
import os
import csv
import io
import time
import random
from typing import Tuple

from loader import ReportLoader
from scorer import GLM4FlashJsonScorer
import local_settings
from timer import Timer   # 你本地的 Timer 实现

# -------------------------
# AI 关键词列表
# -------------------------
AI_KEYWORDS =[
    "人工智能", "AI", "生成式人工智能", "AIGC", "大语言模型", "LLM", "机器学习", "深度学习", "神经网络", "自然语言处理",
    "NLP", "计算机视觉", "图像识别", "语音识别", "语音合成", "知识图谱", "数据挖掘", "预测分析", "算法", "模型",
    "强化学习", "监督学习", "无监督学习", "迁移学习", "联邦学习", "训练数据", "数据标注", "特征工程", "模型训练", "模型优化",
    "模型部署", "推理", "参数", "算力", "GPU", "TPU", "人工智能芯片", "开源模型", "行业模型", "私有模型",
    "智能客服", "聊天机器人", "智能助手", "智能制造", "工业互联网", "智能仓储", "智能物流", "智慧城市", "智慧金融", "智能风控",
    "智能投顾", "智能营销", "精准营销", "个性化推荐", "智慧医疗", "辅助诊断", "智能驾驶", "自动驾驶", "高级辅助驾驶", "智能座舱",
    "智慧能源", "智能电网", "智能家居", "智能安防", "人脸识别", "行为分析", "内容生成", "代码生成", "数字人", "虚拟偶像",
    "数字化转型", "智能化转型", "AI战略", "AI赋能", "AI驱动", "技术中台", "数据中台", "AI平台", "AI云服务", "人机协同",
    "AI伦理", "负责任的人工智能", "AI治理", "数据安全", "算法公平", "AI研发", "AI实验室", "AI人才", "数据科学家", "算法工程师",
    "大数据", "云计算", "物联网", "工业4.0", "机器人流程自动化", "智能机器人", "元宇宙", "区块链", "数字孪生", "智能化"
]

MAX_RETRIES = 5
BASE_SLEEP = 1.0  # base for exponential backoff
# -------------------------

def chunk_contains_ai(chunk: str):
    if not chunk:
        return False
    text = chunk.lower()
    for kw in AI_KEYWORDS:
        if kw.lower() in text:
            return True
    return False


def parse_fname_to_firm_year(fname: str):
    base = os.path.basename(fname)
    parts = base.split("_")
    if len(parts) < 2:
        return None, None
    # 保持 year 字段原样（字符串），后续可转换
    return parts[0], parts[1]


def read_last_csv_line(output_csv: str):
    """
    从 CSV 文件末尾往上查找第一条有效数据行（跳过 header / 空行 / 损坏行）。
    返回 (last_fname, last_chunk_id)；若无数据返回 (None, 0)。
    """
    if not os.path.exists(output_csv):
        return None, 0

    try:
        with open(output_csv, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception as e:
        print(f"[read_last_csv_line] 无法读取文件 {output_csv}: {e}")
        return None, 0

    for raw in reversed(lines):
        line = raw.strip()
        if not line:
            continue
        # 解析 CSV 行以正确处理引号/逗号
        try:
            row = next(csv.reader(io.StringIO(line)))
        except Exception:
            continue
        # 期望至少有 4 列： fname, firm_id, year, chunk_id, ...
        if len(row) < 4:
            continue
        # 如果是 header，则跳过
        first_col = row[0].strip().lower()
        if first_col == "fname":
            continue
        fname = row[0].strip()
        chunk_id_raw = row[3].strip()
        try:
            chunk_id = int(chunk_id_raw)
        except Exception:
            try:
                chunk_id = int(float(chunk_id_raw))
            except Exception:
                continue
        return fname, chunk_id

    return None, 0


def ensure_csv_header(output_csv: str):
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fname", "firm_id", "year", "chunk_id", "chunk_len", "score", "ai_flag"])


def safe_write_row(output_csv: str, row: list):
    """
    追加写一行并 flush + fsync（若可用）
    """
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def run_json_scoring_resume_by_lastline(num_files: None, output_csv: str = "chunk_scores.csv"):
    reading_path = local_settings.YEARLY_REPORTS_PATH
    api_key = local_settings.GLM4_FLASH_API_KEY

    # 更大 overlap 减少关键词被截断风险（如需再调大，修改这里）
    loader = ReportLoader(skip_pages=5, chunk_size=2000, chunk_overlap=500)
    scorer = GLM4FlashJsonScorer(api_key=api_key, model="glm-4-flash")

    # 准备输出 CSV 与断点信息
    ensure_csv_header(output_csv)
    last_processed_file, last_processed_chunk_id = read_last_csv_line(output_csv)
    if last_processed_file:
        print(f"[Resume] last processed file: {last_processed_file}, last chunk_id: {last_processed_chunk_id}")
    else:
        print("[Resume] no previous data found → start from beginning")

    overall_timer = Timer(name="Overall scoring batch")
    overall_timer.start()

    start_processing = True if last_processed_file is None else False

    # iter_files 按文件逐个返回 (fname, chunks)
    for i, (fname, chunks) in enumerate(loader.iter_files(reading_path, num_files=num_files), start=1):
        # 若存在断点且还没到断点文件之前，跳过
        if not start_processing:
            if fname == last_processed_file:
                start_processing = True
                print(f"[Resume] reached last processed file: {fname}, will resume inside this file from chunk {last_processed_chunk_id+1}")
            else:
                # skip this file
                continue

        print(f"\n=== Processing file {i}: {fname} (chunks: {len(chunks)}) ===")
        firm_id, year = parse_fname_to_firm_year(fname)
        if firm_id is None or year is None:
            print(f"[Warning] 无法从文件名解析 firm_id/year: {fname}")

        skip_up_to = last_processed_chunk_id if (last_processed_file and fname == last_processed_file) else 0
        wrote_any_chunk = False  # 本文件是否写入过 ai_flag=1 的 chunk（包含 AI）

        # 遍历 chunks（1-based）
        for idx, chunk in enumerate(chunks, start=1):
            if idx <= skip_up_to:
                continue

            # 关键词预筛：若 chunk 不含 AI 关键词，则跳过评分
            if not chunk_contains_ai(chunk):
                continue

            # 现在是含 AI 的 chunk，需要打分
            wrote_any_chunk = True
            chunk_len = len(chunk)

            # 重试机制
            score = None
            for attempt in range(MAX_RETRIES):
                try:
                    score = scorer.score_chunk(chunk, debug=False)
                    break
                except Exception as e:
                    wait_time = BASE_SLEEP * (2 ** attempt) + random.random()
                    print(f"[Retry {attempt+1}/{MAX_RETRIES}] 打分出错：{fname} chunk-{idx} -> {e}. 等待 {wait_time:.1f}s 后重试...")
                    time.sleep(wait_time)
                    score = None
            if score is None:
                print(f"[Warning] {fname} chunk-{idx} 连续 {MAX_RETRIES} 次失败，记录 score=None 并继续。")

            # 写入行（ai_flag=1）
            row = [fname, firm_id, year, idx, chunk_len, score, 1]
            safe_write_row(output_csv, row)

        # 如果整份报告没有任何含 AI 的 chunk（wrote_any_chunk False），写默认行 ai_flag=0
        if not wrote_any_chunk:
            print(f"[No AI content] {fname} — 写入默认行 ai_flag=0")
            row = [fname, firm_id, year, 0, 0, 0, 0]
            safe_write_row(output_csv, row)

    elapsed = overall_timer.stop()
    print(f"\nTimer 'Overall scoring batch': {elapsed:.4f} seconds")
    print(f"结果写入: {os.path.abspath(output_csv)}")

    # 尝试返回 pandas DataFrame（便于后续处理/调试），若失败则返回 None
    try:
        import pandas as pd
        df = pd.read_csv(output_csv, dtype={"fname": str, "chunk_id": int, "ai_flag": int})
        return df
    except Exception:
        return None


if __name__ == "__main__":
    # 调试时可把 num_files 设为较小值，None 表示全部
    output_path = r"C:\Code\Article\Output\chunk_scores.csv"
    df = run_json_scoring_resume_by_lastline(num_files=20, output_csv=output_path)
    if df is not None:
        print(df.head(10))
