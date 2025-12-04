import re
from pathlib import Path
from collections import Counter

def discover_patterns(filepath, top_n=10, prefix_len=5):
    """
    Naive method deprecated
    
    Analyzes a document to discover and suggest repetitive patterns for removal.

    Args:
        filepath (str): The path to the document file.
        top_n (int): The number of top results to show for frequency counts.
        prefix_len (int): The number of words to consider as a line prefix.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{filepath}'")
        return

    print(f"🔍 Analyzing '{filepath}' to discover removable noise patterns...")
    print("-" * 60)

    # --- 1. Exact Line Analysis ---
    print(f"📊 Top {top_n} Most Frequent Exact Lines:")
    line_counts = Counter(lines)
    for line, count in line_counts.most_common(top_n):
        if count > 1:
            print(f"  - Count: {count:<4} | Line: '{line}'")
    print("-" * 60)

    # --- 2. Prefix Analysis ---
    print(f"📊 Top {top_n} Most Frequent Line Prefixes ({prefix_len} words):")
    prefixes = [' '.join(line.split()[:prefix_len]) for line in lines]
    prefix_counts = Counter(prefixes)
    for prefix, count in prefix_counts.most_common(top_n):
        if count > 1:
            print(f"  - Count: {count:<4} | Prefix: '{prefix}'")
    print("-" * 60)

    # --- 3. Structural Analysis ---
    print("📊 Structural Patterns Found:")
    page_numbers = {line for line in lines if line.isdigit()}
    if page_numbers:
        print(f"  - Found {len(page_numbers)} lines that are just numbers (potential page numbers).")

    html_comments = {line for line in lines if '' in line}
    if html_comments:
        print(f"  - Found {len(html_comments)} lines containing HTML-style comments.")
    print("-" * 60)
    
    print("✅ Analysis complete. Use these suggestions to update your cleaning script.")


if __name__ == "__main__":
    # The document you want to analyze
    document_to_analyze = Path("data/test/examples.md")
    discover_patterns(document_to_analyze)