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
    def __init__(self, skip_pages = 5, chunk_size = 1000, chunk_overlap = 200):
        self.skip_pages = skip_pages
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.failed_to_read = []

    def extract_sections(self, text, context_window = 200):
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

    def load_and_chunk(self, pdf_path):
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

    def iter_files(self, pdf_dir, num_files = None):
        all_files = sorted(
            f for f in os.listdir(pdf_dir)
            if os.path.isfile(os.path.join(pdf_dir, f)) and f.lower().endswith(".pdf")
        )
        if num_files:
            all_files = all_files[:num_files]


        for fname in all_files:
            path = os.path.join(pdf_dir, fname)
            chunks = self.load_and_chunk(path)
            yield fname, chunks


if __name__ == "__main__":
    pdf_folder = r"C:\Code\Article\Spider\scrape-cop-reports-CnInfo\YearlyReport\A股年报"  # 修改为你的PDF目录
    loader = ReportLoader(skip_pages=5, chunk_size=1000, chunk_overlap=200)

    # 只读取前 2 份PDF用于测试
    for fname, chunks in loader.iter_files(pdf_folder, num_files=2):
        print(f"\n--- {fname}: 共 {len(chunks)} 个chunk")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i} 内容:\n{'='*40}\n{chunk}\n{'='*40}")

    if loader.failed_to_read:
        print("\n未能成功提取的文件:", loader.failed_to_read)
    else:
        print("\n所有选定文件均已成功处理。")
