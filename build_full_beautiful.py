#!/usr/bin/env python3
"""
Build the complete TKS v7.4 Beautiful Edition by combining:
1. The beautiful preamble from TKS_v7_4_BEAUTIFUL.tex
2. The converted content from TKS_v7_4_CONVERTED.tex
3. The corrected chapters (27 and 52-54)
"""

import re

def main():
    # Read the beautiful preamble (everything up to \begin{document})
    with open('TKS_v7_4_BEAUTIFUL.tex', 'r') as f:
        beautiful = f.read()
    
    # Extract preamble
    preamble_end = beautiful.find('\\begin{document}')
    preamble = beautiful[:preamble_end]
    
    # Read the converted body
    with open('converted_body.tex', 'r') as f:
        body = f.read()
    
    # Clean up the body - remove the old title/toc since we have new ones
    body = re.sub(r'\\maketitle\s*', '', body)
    body = re.sub(r'\\tableofcontents\s*', '', body)
    
    # Build the complete document
    output = []
    output.append(preamble)
    output.append('\\begin{document}\n')
    output.append('\\maketitle\n')
    output.append('\\frontmatter\n')
    
    # Add canonical authority declaration
    output.append('''
%% ============================================================================
%% CANONICAL AUTHORITY DECLARATION
%% ============================================================================
\\chapter*{Canonical Authority Declaration}
\\addcontentsline{toc}{chapter}{Canonical Authority Declaration}

\\infobox{v7.3 $\\to$ v7.4 Continuity Bridge}{
\\textbf{Canonical Inheritance Declaration.} This document, TKS v7.4, is a \\emph{strict superset} of TKS v7.3, which itself preserves all of v7.2, v7.1, v7.0, and v6.1. All definitions, theorems, axioms, and proofs from previous versions remain valid and unchanged. Version 7.4 introduces four new domain expansions---no modifications to existing canon.
}

\\textbf{Canonical Status of Version 7.4.} This document constitutes the canonical and authoritative specification of the TKS Formal Mathematical System. Version 7.4 fully integrates, refines, and supersedes all prior releases in the v3.x--v7.3 development line.

The present release establishes the \\textbf{definitive ontology of TKS}, consisting of:
\\begin{itemize}
    \\item The forty canonical Elements (A1--D10)
    \\item The ten Noetic operators $\\{\\nu_0, \\ldots, \\nu_9\\}$
    \\item The seven Foundations and their twenty-eight Sub-Foundations
    \\item The twenty-two Acquisitions (A0 together with the D, W, and P chains)
\\end{itemize}

\\tableofcontents

\\mainmatter

''')
    
    # Add the converted body content (skip the front matter stuff we already added)
    # Find where the actual content starts (after the intro material)
    content_start = body.find('\\chapter{')
    if content_start == -1:
        content_start = body.find('Chapter 1')
    
    if content_start > 0:
        output.append(body[content_start:])
    else:
        output.append(body)
    
    output.append('\n\\end{document}\n')
    
    # Write the complete document
    with open('TKS_v7_4_FULL_BEAUTIFUL.tex', 'w') as f:
        f.write(''.join(output))
    
    print("Complete document written to TKS_v7_4_FULL_BEAUTIFUL.tex")

if __name__ == '__main__':
    main()
