"""
preprocess.py

Purpose:
- Take a scripture PDF as input
- Extract text
- Normalize text
- Sentence-split
- Chunk into LLM-friendly sizes
- Write structured CSV (one row per chunk)
"""

import os
import sys
import csv
import argparse
import re
from typing import List

MAX_WORDS_PER_CHUNK = 200


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")

    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)

    return "\n".join(full_text)


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines

def sentence_split(text: str) -> List[str]:
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
    except ImportError:
        raise RuntimeError("nltk not installed. Run: pip install nltk")

    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)


    return sent_tokenize(text)


def chunk_sentences(sentences: List[str], max_words: int) -> List[str]:
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        wc = len(sentence.split())

        if current_word_count + wc <= max_words:
            current_chunk.append(sentence)
            current_word_count += wc
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = wc

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def write_csv(output_path: str, scripture: str, source_pdf: str, chunks: List[str]):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scripture",
                "source_pdf",
                "chunk_id",
                "chunk_index",
                "text"
            ]
        )
        writer.writeheader()

        for idx, chunk in enumerate(chunks):
            writer.writerow({
                "scripture": scripture,
                "source_pdf": source_pdf,
                "chunk_id": f"{scripture}_C{idx:04d}",
                "chunk_index": idx,
                "text": chunk
            })


def main():
    parser = argparse.ArgumentParser(description="Prepare scripture PDF into chunked CSV")
    parser.add_argument("--pdf", required=True, help="Path to scripture PDF")
    parser.add_argument("--scripture", required=True, help="GITA | QURAN | BIBLE")
    parser.add_argument("--out", required=True, help="Output CSV path")

    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"ERROR: PDF not found: {args.pdf}")
        sys.exit(1)

    print("[1/5] Extracting text from PDF...")
    raw_text = extract_text_from_pdf(args.pdf)

    print("[2/5] Normalizing text...")
    clean_text = normalize_text(raw_text)

    print("[3/5] Sentence splitting (line-aware)...")
    sentences = []

    for line in clean_text:
        if re.search(r"[.!?]", line):
            line_sentences = sentence_split(line)
            sentences.extend(line_sentences)
        else:
            sentences.append(line)

    print(f"    Total sentences: {len(sentences)}")

    print("[4/5] Chunking sentences...")
    chunks = chunk_sentences(sentences, MAX_WORDS_PER_CHUNK)
    print(f"    Total chunks: {len(chunks)}")

    print("[5/5] Writing CSV...")
    write_csv(
        output_path=args.out,
        scripture=args.scripture,
        source_pdf=os.path.basename(args.pdf),
        chunks=chunks
    )

    print("DONE.")


if __name__ == "__main__":
    main()

