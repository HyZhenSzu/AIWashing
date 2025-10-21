"""
aggregate_scores.py

聚合 chunk-level 得分为 firm-year 级别指标。
输出：aggregated_scores.csv
"""

import os
import pandas as pd
import numpy as np


INPUT_CSV = r"C:\Code\Article\Output\chunk_scores.csv"
OUTPUT_CSV = r"C:\Code\Article\Output\aggregated_scores.csv"
TOP_K = 3  # top-k 平均


def aggregate_firm_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入 chunk-level DataFrame，返回 firm-year 聚合结果
    """
    grouped = []
    for (fname, firm_id, year), g in df.groupby(["fname", "firm_id", "year"]):
        # 只取含 AI 的 chunk
        g_ai = g[g["ai_flag"] == 1]
        if g_ai.empty:
            grouped.append({
                "fname": fname,
                "firm_id": firm_id,
                "year": year,
                "weighted_avg_score": 0,
                "max_score": 0,
                "topk_avg_score": 0,
                "median_score": 0,
                "q75_score": 0,
                "q90_score": 0,
                "z_score_based": 0,
                "presence_dummy": 0,
                "chunk_count": 0,
                "total_len": 0
            })
            continue

        scores = g_ai["score"].astype(float).fillna(0).values
        lengths = g_ai["chunk_len"].astype(float).fillna(0).values

        # --- 主指标：加权平均 ---
        weighted_avg = np.average(scores, weights=lengths) if lengths.sum() > 0 else 0

        # --- 其他指标 ---
        max_score = float(np.max(scores))
        topk_avg = float(np.mean(sorted(scores, reverse=True)[:TOP_K]))
        median_score = float(np.median(scores))
        q75_score = float(np.quantile(scores, 0.75))
        q90_score = float(np.quantile(scores, 0.90))

        # --- z-score based threshold ---
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            z_based = 0
        else:
            z_scores = (scores - mean) / std
            z_based = 1 if np.mean(z_scores > 1) > 0 else 0

        presence_dummy = 1
        chunk_count = len(g_ai)
        total_len = np.sum(lengths)

        # --- 四舍五入到小数点后两位 ---
        grouped.append({
            "fname": fname,
            "firm_id": firm_id,
            "year": year,
            "weighted_avg_score": round(weighted_avg, 2),
            "max_score": round(max_score, 2),
            "topk_avg_score": round(topk_avg, 2),
            "median_score": round(median_score, 2),
            "q75_score": round(q75_score, 2),
            "q90_score": round(q90_score, 2),
            "z_score_based": z_based,
            "presence_dummy": presence_dummy,
            "chunk_count": chunk_count,
            "total_len": total_len
        })

    return pd.DataFrame(grouped)


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"未找到输入文件: {INPUT_CSV}")

    print(f"正在加载 {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"fname", "firm_id", "year", "chunk_id", "chunk_len", "score", "ai_flag"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"输入文件缺少必要列: {required_cols - set(df.columns)}")

    print(f"共 {len(df)} 条 chunk-level 记录，将按 firm_id/year 聚合。")
    agg_df = aggregate_firm_year(df)
    agg_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n聚合完成，输出文件: {OUTPUT_CSV}")
    print(agg_df.head())


if __name__ == "__main__":
    main()
