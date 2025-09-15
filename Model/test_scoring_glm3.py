from loader import ReportLoader  # 你已有的 ReportLoader
from scorer_glm3 import LLMSCorer

READING_PATH = r"C:\Code\Article\Spider\scrape-cop-reports-CnInfo\YearlyReport\A股年报"

loader = ReportLoader(skip_pages=5, chunk_size=2000, chunk_overlap=300)
scorer = LLMSCorer()

results = loader.process_dir(READING_PATH, num_files=10)
for fname, chunks in results.items():
    scores = [scorer.score_chunk_debug(c) for c in chunks]
    overall = sum(scores) / len(scores) if scores else None
    print(f"{fname}: chunk scores={scores}, overall={overall:.2f}")
