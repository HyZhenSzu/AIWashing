from loader import ReportLoader
from scorer import GLM4FlashJsonScorer
import local_settings as local_settings
from timer import Timer

def run_json_scoring():
    reading_path = local_settings.YEARLY_REPORTS_PATH
    api_key = local_settings.GLM4_FLASH_API_KEY

    loader = ReportLoader(skip_pages=5, chunk_size=2000, chunk_overlap=300)
    scorer = GLM4FlashJsonScorer(api_key=api_key, model="glm-4-flash")

    overall_timer = Timer(name="Overall scoring for batch")
    overall_timer.start()

    results = loader.iter_files(reading_path, num_files=10)
    for fname, chunks in results.items():
        file_timer = Timer(name=f"Scoring file {fname}")
        file_timer.start()

        scores = []
        for c in chunks:
            s = scorer.score_chunk(c, debug=True)
            scores.append(s)
        overall_file_time = file_timer.stop()
        overall = sum(scores) / len(scores) if scores else None
        print(f"{fname} → chunk scores={scores}, overall={overall:.2f}")
        print(f"Time to score {fname}: {overall_file_time:.2f} seconds\n")
    
    total_time = overall_timer.stop()
    print(f"Total time to score all files: {total_time:.2f} seconds")

if __name__ == "__main__":
    run_json_scoring()