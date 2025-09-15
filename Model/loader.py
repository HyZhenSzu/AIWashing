import os
import re
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

HEADING_VARIANTS = [
    "管理层讨论与分析",
    "公司业务概要",
    "经营情况讨论与分析",
]

END_TITLES = r"(公司治理|环境和社会责任|重要事项|股份变动及股东情况)"

class ReportLoader:
    def __init__(self, skip_pages: int = 5, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.skip_pages = skip_pages
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.failed_to_read: List[str] = []

    def extract_sections(self, text: str, context_window: int = 200) -> str:
        sections = []
        for heading in HEADING_VARIANTS:
            for match in re.finditer(heading, text):
                idx = match.start()
                snippet = text[max(0, idx - context_window): idx + context_window]
                if re.search(r"\.{3,}", snippet) or re.search(r"第?\d+页", snippet):
                    continue
                after = text[idx:]
                end_match = re.search(END_TITLES, after)
                end_idx = idx + (end_match.start() if end_match else len(after))
                sections.append(text[idx:end_idx])
                break
        return "\n".join(sections) if sections else ""

    def load_and_chunk(self, pdf_path: str) -> List[str]:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()[self.skip_pages:]
        full_text = "\n".join(p.page_content for p in pages)

        if not any(h in full_text for h in HEADING_VARIANTS):
            self.failed_to_read.append(os.path.basename(pdf_path))
            return []

        content = self.extract_sections(full_text)
        if not content:
            self.failed_to_read.append(os.path.basename(pdf_path))
            return []

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(content)

    def process_dir(self, pdf_dir: str, num_files: int = None) -> Dict[str, List[str]]:
        results = {}
        # 推荐使用 os.scandir() 或 sorted list from os.listdir()
        all_files = sorted(f for f in os.listdir(pdf_dir) if os.path.isfile(os.path.join(pdf_dir, f)) and f.lower().endswith(".pdf"))

        if num_files:
            all_files = all_files[:num_files]

        for fname in all_files:
            path = os.path.join(pdf_dir, fname)
            chunks = self.load_and_chunk(path)
            results[fname] = chunks

        return results



if __name__ == "__main__":
    pdf_folder = r"C:\Code\Article\Spider\scrape-cop-reports-CnInfo\YearlyReport\A股年报"
    loader = ReportLoader(skip_pages=5, chunk_size=3000, chunk_overlap=500)

    # 只读取前 3 份年报，用于快速测试
    processed = loader.process_dir(pdf_folder, num_files=3)

    for fname, chunks in processed.items():
        print(f"\n--- {fname}: found {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            first_line = chunk.strip().split("\n")[0]
            print(f"  Chunk {i} first line: {first_line[:50]}... length: {len(chunk)} chars")

    if loader.failed_to_read:
        print("\nFailed to extract sections from:", loader.failed_to_read)
    else:
        print("\nAll selected files processed successfully.")
