#!/usr/bin/env python3
"""
TKS v7.4 Full LaTeX Converter v2
Converts raw alltt blocks to proper LaTeX formatting with properly closed theorem environments.
"""

import re
import sys

def fix_ligatures(text):
    """Fix common OCR ligature issues."""
    # First fix the common ligature patterns with single quote
    text = re.sub(r"([a-z])'([a-z])", r"\1fi\2", text)  # x'y -> xfiy
    text = re.sub(r"([A-Z])'([a-z])", r"\1fi\2", text)  # X'y -> Xfiy
    
    replacements = [
        ("''", "fi"),
        ("de'nition", "definition"),
        ("De'nition", "Definition"),
        ("'nite", "finite"),
        ("'xed", "fixed"),
        ("'lter", "filter"),
        ("uni'cation", "unification"),
        ("speci'cation", "specification"),
        ("re'ne", "refine"),
        ("codi'ed", "codified"),
        ("in'uence", "influence"),
        ("'eld", "field"),
        ("satis'", "satisfi"),
        ("veri'", "verifi"),
        ("classi'", "classifi"),
        ("'rst", "first"),
        ("aect", "affect"),
        ("eect", "effect"),
        ("dierent", "different"),
        ("dierential", "differential"),
        ("Hausdor", "Hausdorff"),
        ("con'guration", "configuration"),
        ("'bration", "fibration"),
        ("'ber", "fiber"),
        ("mani'old", "manifold"),
        ("'nal", "final"),
        ("in'nite", "infinite"),
        ("in'mum", "infimum"),
        ("'x", "fix"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def fix_latex_escapes(text):
    """Fix escaped LaTeX commands from alltt."""
    text = re.sub(r'\\textbackslash\\\{\\\}(\w+)', r'\\\1', text)
    text = text.replace("\\{", "{")
    text = text.replace("\\}", "}")
    text = text.replace("\\_", "_")
    return text

def convert_environments_with_closing(text):
    """
    Convert theorem-like environments and properly close them.
    This uses a state machine approach to track open environments.
    """
    # Environment patterns (start markers)
    env_patterns = [
        (r"Definition\s+(\d+\.\d+)\s*\(([^)]+)\)[.:]?", "definition", "def"),
        (r"Theorem\s+(\d+\.\d+)\s*\(([^)]+)\)[.:]?", "theorem", "thm"),
        (r"Lemma\s+(\d+\.\d+)\s*\(([^)]+)\)[.:]?", "lemma", "lem"),
        (r"Proposition\s+(\d+\.\d+)\s*\(([^)]+)\)[.:]?", "proposition", "prop"),
        (r"Corollary\s+(\d+\.\d+)\s*\(([^)]+)\)[.:]?", "corollary", "cor"),
        (r"Remark\s+(\d+\.\d+)\s*\(([^)]+)\)[.:]?", "remark", "rem"),
        (r"Example\s+(\d+\.\d+)\s*\(([^)]+)\)[.:]?", "example", "ex"),
        (r"Axiom\s+(\d+\.\d+)\s*\(([^)]+)\)[.:]?", "axiom", "ax"),
        (r"Construction\s+(\d+\.\d+)\s*\(([^)]+)\)[.:]?", "construction", "constr"),
        (r"Notation\s+(\d+\.\d+)\s*\(([^)]+)\)[.:]?", "notation", "not"),
    ]
    
    # Section patterns that close environments
    section_patterns = [
        r"^Chapter\s+\d+",
        r"^\d+\.\d+\s+[A-Z]",  # Section headers like "2.3 Title"
        r"^\d+\.\d+\.\d+\s+[A-Z]",  # Subsection headers
        r"^\\chapter\{",
        r"^\\section\{",
        r"^\\subsection\{",
        r"^\\clearpage",
    ]
    
    lines = text.split('\n')
    result = []
    current_env = None  # Track the currently open environment
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check if we need to close current environment
        if current_env:
            # Check for section breaks
            should_close = False
            for pattern in section_patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    should_close = True
                    break
            
            # Check for new environment start
            if not should_close:
                for pattern, env_name, _ in env_patterns:
                    if re.match(pattern, stripped, re.IGNORECASE):
                        should_close = True
                        break
            
            # Check for Proof start (closes previous env)
            if re.match(r"^Proof\.", stripped):
                should_close = True
            
            if should_close:
                result.append(f"\\end{{{current_env}}}\n")
                current_env = None
        
        # Check for environment start
        env_started = False
        for pattern, env_name, label_prefix in env_patterns:
            match = re.match(pattern, stripped, re.IGNORECASE)
            if match:
                num = match.group(1)
                name = match.group(2).strip()
                # Close any previous environment first
                if current_env:
                    result.append(f"\\end{{{current_env}}}\n")
                result.append(f"\n\\begin{{{env_name}}}[{name}]")
                result.append(f"\\label{{{label_prefix}:{num.replace('.', '-')}}}")
                current_env = env_name
                # Add remaining content after the match
                remaining = stripped[match.end():].strip()
                if remaining:
                    result.append(remaining)
                env_started = True
                break
        
        if not env_started:
            result.append(line)
    
    # Close any remaining open environment
    if current_env:
        result.append(f"\\end{{{current_env}}}\n")
    
    return '\n'.join(result)

def convert_chapters(text):
    """Convert Chapter N headers to LaTeX."""
    pattern = r"^Chapter\s+(\d+)\s*\n+([A-Z][^\n]+)"
    
    def replace_ch(match):
        num = match.group(1)
        title = match.group(2).strip()
        return f"\n\\chapter{{{title}}}\n\\label{{ch:{num}}}\n"
    
    text = re.sub(pattern, replace_ch, text, flags=re.MULTILINE)
    return text

def convert_sections(text):
    """Convert X.Y Section Title patterns to LaTeX."""
    pattern = r"^(\d+\.\d+)\s+([A-Z][A-Za-z\s\-]+)$"
    
    def replace_sec(match):
        num = match.group(1)
        title = match.group(2).strip()
        if any(word in title for word in ['Definition', 'Theorem', 'Lemma', 'Proposition']):
            return match.group(0)
        return f"\n\\section{{{title}}}\n\\label{{sec:{num.replace('.', '-')}}}\n"
    
    text = re.sub(pattern, replace_sec, text, flags=re.MULTILINE)
    return text

def convert_subsections(text):
    """Convert X.Y.Z Subsection Title patterns to LaTeX."""
    pattern = r"^(\d+\.\d+\.\d+)\s+([A-Z][A-Za-z\s\-\(\)]+)$"
    
    def replace_subsec(match):
        num = match.group(1)
        title = match.group(2).strip()
        return f"\n\\subsection{{{title}}}\n\\label{{subsec:{num.replace('.', '-')}}}\n"
    
    text = re.sub(pattern, replace_subsec, text, flags=re.MULTILINE)
    return text

def convert_proofs(text):
    """Convert Proof. to LaTeX proof environment."""
    # Find Proof. and add \begin{proof}
    text = re.sub(r"^Proof\.\s*", "\n\\begin{proof}\n", text, flags=re.MULTILINE)
    
    # Try to close proofs at common endings
    # This is heuristic - proofs often end with □ or before the next theorem
    text = re.sub(r"□\s*\n", r"\\end{proof}" + "\n", text)
    
    return text

def remove_page_numbers(text):
    """Remove standalone page numbers."""
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text

def remove_chapter_headers(text):
    """Remove redundant chapter headers like 'CHAPTER 1. TITLE'."""
    text = re.sub(r"^CHAPTER\s+\d+\.\s+[A-Z\s]+$", "", text, flags=re.MULTILINE)
    return text

def process_alltt_block(text):
    """Process a single alltt block."""
    text = fix_ligatures(text)
    text = fix_latex_escapes(text)
    text = remove_page_numbers(text)
    text = remove_chapter_headers(text)
    text = convert_chapters(text)
    text = convert_sections(text)
    text = convert_subsections(text)
    text = convert_environments_with_closing(text)
    text = convert_proofs(text)
    
    # Clean up multiple blank lines
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    return text

def main():
    """Main conversion function."""
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'TKS_v7_4_FULL_CORRECTED.tex'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'TKS_v7_4_CONVERTED.tex'
    
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Find alltt blocks
    alltt_pattern = r'\\begin\{alltt\}(.*?)\\end\{alltt\}'
    matches = list(re.finditer(alltt_pattern, content, re.DOTALL))
    
    print(f"Found {len(matches)} alltt blocks")
    
    # Process each block
    new_content = content
    offset = 0
    
    for i, match in enumerate(matches):
        start = match.start() + offset
        end = match.end() + offset
        block_content = match.group(1)
        
        print(f"Processing block {i+1} ({len(block_content)} chars)...")
        
        converted = process_alltt_block(block_content)
        
        replacement = f"\n% === CONVERTED FROM ALLTT BLOCK {i+1} ===\n{converted}\n% === END BLOCK {i+1} ===\n"
        
        new_content = new_content[:start] + replacement + new_content[end:]
        offset += len(replacement) - (end - start)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Conversion complete. Output written to {output_file}")
    print(f"Original: {len(content)} chars, Converted: {len(new_content)} chars")

if __name__ == '__main__':
    main()
