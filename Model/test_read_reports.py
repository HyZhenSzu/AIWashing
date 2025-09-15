import os
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

HEADING_VARIANTS = ["管理层讨论与分析", "公司业务概要", "经营情况讨论与分析"]
failed_to_read = [] # 用于记录无法读取的文件

# 优化后的提取函数，支持多个标题起点
def extract_sections(text: str, context_window: int = 200) -> str:
    # 多种可能的起始标题
    
    # 下一章节可能的结束标题集合
    end_titles = r"(公司治理|环境和社会责任|重要事项|股份变动及股东情况)"

    sections = []
    for heading in HEADING_VARIANTS:
        for match in re.finditer(heading, text):
            idx = match.start()
            snippet = text[max(0, idx - context_window): idx + context_window]
            # 过滤目录式匹配，如省略号或页码
            if re.search(r"\.{3,}", snippet) or re.search(r"第?\d+页", snippet):
                continue
            # 定位结束位置
            after = text[idx:]
            end_match = re.search(end_titles, after)
            end_idx = idx + (end_match.start() if end_match else len(after))
            content = text[idx:end_idx]
            sections.append(content)
            break  # 每个标题只提取一次

    if not sections:
        return ""
    return "\n".join(sections)

def process_reports(pdf_dir: str, num_files: int = None,
                    skip_pages: int = 5, chunk_size: int = 1000, chunk_overlap: int = 200):
    filenames = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    if num_files:
        filenames = filenames[:num_files]

    for fname in filenames:
        print(f"\n--- Processing file: {fname} ---")
        path = os.path.join(pdf_dir, fname)
        loader = PyPDFLoader(path)
        pages = loader.load()[skip_pages:]
        text = "\n".join(page.page_content for page in pages)

        if not any(h in text for h in HEADING_VARIANTS):
            print("  → No applicable heading found in body, skip")
            failed_to_read.append(fname)
            continue

        content = extract_sections(text)
        if not content:
            print("  → Failed to extract content from headings")
            continue

        print(f"Extracted text length: {len(content)} chars")

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(content)

        for i, chunk in enumerate(chunks, 1):
            first_line = chunk.strip().split("\n")[0]
            print(f"  Chunk {i} first line: {first_line}")
            print(f"    Length: {len(chunk)} chars")

if __name__ == "__main__":
    pdf_folder = r"C:\Code\Article\Spider\scrape-cop-reports-CnInfo\YearlyReport\A股年报"
    process_reports(pdf_folder, num_files=None, chunk_size=3000, chunk_overlap=500)
    print(f"\nFailed to read files: {failed_to_read}") if failed_to_read else print("All files processed successfully.")
