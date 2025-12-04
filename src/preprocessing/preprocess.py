import re
import os
import sys
from pathlib import Path

# Add src to path so we can import our modules
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))
from utils.logging_config import setup_logger

# Get a logger instance
logger = setup_logger("preprocess")

def step1_clean_noise(lines):
    """
    Line-by-line cleanup to remove universal noise like headers, footers,
    page numbers, HTML comments and proprietary notices.
    """
    cleaned_lines = []
    # Regex-based noise patterns that can appear anywhere in the line
    noise_regex = [
        re.compile(r'PROPRIETARY.*CONFIDENTIAL', re.I),   # covers "&" and "&amp;"
        re.compile(r'^\s*<!-'),                           # malformed comments fragments
    ]

    for line in lines:
        stripped = line.strip()

        # keep blank lines (useful for Markdown readability)
        if stripped == "":
            cleaned_lines.append(line)
            continue

        # remove page numbers (standalone digits)
        if stripped.isdigit():
            continue

        # remove any of the noise patterns
        if any(p.search(line) for p in noise_regex):
            continue

        cleaned_lines.append(line)

    return cleaned_lines


def step2_filter_content(lines):
    """
    Keep:
      - Requirement sections when the requirement ID appears in a markdown heading (e.g. '## HAA-21 / ...')
      - Requirement sections when a line itself starts with an ID (e.g. 'HAA-21 / ...')
      - Also keep non-requirement document headings (e.g. '# Title: ...')
    """
    final_lines = []
    i, n = 0, len(lines)

    req_heading = re.compile(r'^\s{0,3}#{1,6}\s+[A-Z]{2,10}-\d+\b')
    req_line    = re.compile(r'^\s*[A-Z]{2,10}-\d+\b')

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Keep non-requirement headings (skip generic noise headings if desired)
        if stripped.startswith('#') and not req_heading.match(stripped):
            if stripped.lower() in {'# juice', '## juice'}:
                i += 1
                continue
            final_lines.append(line)
            i += 1
            continue

        # Requirement as a heading -> capture until next heading
        if req_heading.match(stripped):
            block = [line]
            i += 1
            while i < n and not lines[i].lstrip().startswith('#'):
                block.append(lines[i])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            final_lines.extend(block + ['\n'])
            continue

        # Requirement as a plain line -> capture until next heading or next requirement
        if req_line.match(stripped):
            block = [line]
            i += 1
            while i < n:
                nxt = lines[i]
                if nxt.lstrip().startswith('#') or req_line.match(nxt.strip()):
                    break
                block.append(nxt)
                i += 1
            while block and not block[-1].strip():
                block.pop()
            final_lines.extend(block + ['\n'])
            continue

        # Otherwise skip
        i += 1

    return final_lines


def main():
    """
    Main function to orchestrate the two-step preprocessing operation.
    """
    #input_filename = Path("data/test/examples.md")
    #output_filename = Path("data/test/cleaned_requirements.md")
    input_dir = Path("data/docling_markdown_output")
    output_dir = Path("data/cleaned_markdown")

    if not os.path.exists(input_dir):
        logger.error(f"❌ Error: Input directory '{input_dir}' not found.")
        return
    
    if not os.path.exists(output_dir):
        logger.error(f"❌ Error: Output directory '{output_dir}' not found.")
        return
    
    # Find all markdown files in input directory
    md_files = list(input_dir.glob("*.md"))

    if not md_files:
        logger.warning(f"No markdown files found in {input_dir} directory")
        return

    logger.info(f"Found {len(md_files)} md file(s) to clean")

    # Process each Md file
    for md_file in md_files:
        try:    
            logger.info(f"📖 Reading from '{md_file.name}'...")
            with open(md_file, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()

            # --- Step 1 ---
            logger.info("🧹 Step 1: Cleaning universal noise from the document...")
            lines_after_step1 = step1_clean_noise(original_lines)
            
            # --- Step 2 ---
            logger.info("🔍 Step 2: Filtering to preserve headings and requirement blocks...")
            final_content_lines = step2_filter_content(lines_after_step1)
            
            # --- Writing Output ---
            output_filename = md_file.stem + ".md"
            output_path = output_dir / output_filename

            logger.info(f"✍️ Writing cleaned content to '{output_path}'...")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(final_content_lines)
                
            logger.info(f"\n✅ Preprocessing complete! ✨")
            logger.info(f"The cleaned file has been saved as '{output_filename}'.")
        except Exception as e:
            logger.error(f"Failed to clean {md_file.name}: {e}")
            print(f"Failed to clean {md_file.name}: {str(e)}")

if __name__ == "__main__":
    main()