# run_scoring_parallel.py
from loader import ReportLoader
from scorer import GLM4FlashJsonScorer
import local_settings
from timer import Timer 
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_json_scoring_parallel():
    reading_path = local_settings.YEARLY_REPORTS_PATH
    api_key = local_settings.GLM4_FLASH_API_KEY

    loader = ReportLoader(skip_pages=5, chunk_size=2000, chunk_overlap=300)
    scorer = GLM4FlashJsonScorer(api_key=api_key, model="glm-4-flash")

    results = loader.process_dir(reading_path, num_files=10)

    overall_timer = Timer(name="Overall scoring parallel batch")
    overall_timer.start()

    for fname, chunks in results.items():
        file_timer = Timer(name=f"Scoring {fname} in parallel")
        file_timer.start()

        scores = []
        # 用线程池来并发处理 chunk
        max_workers = 10  # 你可以调这个，看 API 能承受多少并发
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(scorer.score_chunk, c, False): c for c in chunks}
            for future in as_completed(future_to_chunk):
                chunk_text = future_to_chunk[future]
                try:
                    s = future.result()
                except Exception as e:
                    print(f"Error scoring chunk in file {fname}: {e}")
                    s = None
                scores.append(s)

        file_time = file_timer.stop()
        # 如果有 None 值，可以过滤掉
        valid_scores = [s for s in scores if s is not None]
        overall = sum(valid_scores) / len(valid_scores) if valid_scores else None
        print(f"{fname}: chunk scores={valid_scores}, overall={overall:.2f}" if overall is not None else f"{fname}: no valid scores")
        print(f"Time for file {fname}: {file_time:.2f}s\n")

    total_time = overall_timer.stop()
    print(f"Total time for batch: {total_time:.2f}s")

if __name__ == "__main__":
    run_json_scoring_parallel()
