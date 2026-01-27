#!/usr/bin/env python3
"""
Fix Unicode characters in LaTeX file by converting them to LaTeX commands.
"""

import re

def fix_unicode(text):
    """Replace Unicode math symbols with LaTeX equivalents."""
    # First remove control characters
    import unicodedata
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cc' or c in '\n\r\t')
    
    # Fix common OCR artifacts
    text = text.replace('^^', 'ffi')  # Common ligature artifact
    text = text.replace('e^^', 'effi')  # efficiency, coefficient, etc.
    replacements = [
        # Greek letters
        ('α', r'$\alpha$'),
        ('β', r'$\beta$'),
        ('γ', r'$\gamma$'),
        ('δ', r'$\delta$'),
        ('ε', r'$\epsilon$'),
        ('ζ', r'$\zeta$'),
        ('η', r'$\eta$'),
        ('θ', r'$\theta$'),
        ('ι', r'$\iota$'),
        ('κ', r'$\kappa$'),
        ('λ', r'$\lambda$'),
        ('μ', r'$\mu$'),
        ('ν', r'$\nu$'),
        ('ξ', r'$\xi$'),
        ('π', r'$\pi$'),
        ('ρ', r'$\rho$'),
        ('σ', r'$\sigma$'),
        ('τ', r'$\tau$'),
        ('υ', r'$\upsilon$'),
        ('φ', r'$\phi$'),
        ('χ', r'$\chi$'),
        ('ψ', r'$\psi$'),
        ('ω', r'$\omega$'),
        ('ϵ', r'$\epsilon$'),
        ('ϕ', r'$\phi$'),
        ('ϑ', r'$\vartheta$'),
        ('ϱ', r'$\varrho$'),
        ('ς', r'$\varsigma$'),
        ('ϰ', r'$\varkappa$'),
        ('Γ', r'$\Gamma$'),
        ('Δ', r'$\Delta$'),
        ('Θ', r'$\Theta$'),
        ('Λ', r'$\Lambda$'),
        ('Ξ', r'$\Xi$'),
        ('Π', r'$\Pi$'),
        ('Σ', r'$\Sigma$'),
        ('Φ', r'$\Phi$'),
        ('Ψ', r'$\Psi$'),
        ('Ω', r'$\Omega$'),
        
        # Math operators and relations
        ('∈', r'$\in$'),
        ('∉', r'$\notin$'),
        ('⊂', r'$\subset$'),
        ('⊃', r'$\supset$'),
        ('⊆', r'$\subseteq$'),
        ('⊇', r'$\supseteq$'),
        ('⊊', r'$\subsetneq$'),
        ('⊋', r'$\supsetneq$'),
        ('⊑', r'$\sqsubseteq$'),
        ('⊒', r'$\sqsupseteq$'),
        ('⊓', r'$\sqcap$'),
        ('⊔', r'$\sqcup$'),
        ('∪', r'$\cup$'),
        ('∩', r'$\cap$'),
        ('∅', r'$\emptyset$'),
        ('∀', r'$\forall$'),
        ('∃', r'$\exists$'),
        ('¬', r'$\neg$'),
        ('∧', r'$\wedge$'),
        ('∨', r'$\vee$'),
        ('⊕', r'$\oplus$'),
        ('⊗', r'$\otimes$'),
        ('⊖', r'$\ominus$'),
        ('⊙', r'$\odot$'),
        ('∗', r'$*$'),
        ('⋆', r'$\star$'),
        ('⋅', r'$\cdot$'),
        ('⋮', r'$\vdots$'),
        ('⋯', r'$\cdots$'),
        ('⋰', r'$\ddots$'),
        ('⋱', r'$\ddots$'),
        
        # Arrows
        ('→', r'$\to$'),
        ('←', r'$\leftarrow$'),
        ('↔', r'$\leftrightarrow$'),
        ('⇒', r'$\Rightarrow$'),
        ('⇐', r'$\Leftarrow$'),
        ('⇔', r'$\Leftrightarrow$'),
        ('⇑', r'$\Uparrow$'),
        ('⇓', r'$\Downarrow$'),
        ('⇕', r'$\Updownarrow$'),
        ('↦', r'$\mapsto$'),
        ('7→', r'$\mapsto$'),
        ('↪', r'$\hookrightarrow$'),
        ('↠', r'$\twoheadrightarrow$'),
        
        # Comparison
        ('≤', r'$\leq$'),
        ('≥', r'$\geq$'),
        ('≠', r'$\neq$'),
        ('≈', r'$\approx$'),
        ('≡', r'$\equiv$'),
        ('≃', r'$\simeq$'),
        ('≅', r'$\cong$'),
        ('∼', r'$\sim$'),
        ('≪', r'$\ll$'),
        ('≫', r'$\gg$'),
        ('⪯', r'$\preceq$'),
        ('⪰', r'$\succeq$'),
        ('≺', r'$\prec$'),
        ('≻', r'$\succ$'),
        ('̄', ''),  # combining macron
        ('̅', ''),  # combining overline
        ('̂', ''),  # combining circumflex
        ('̃', ''),  # combining tilde
        ('̇', ''),  # combining dot above
        ('̈', ''),  # combining diaeresis
        ('̧', ''),  # combining cedilla
        ('̸=', r'$\neq$'),
        
        # Proportional
        ('∝', r'$\propto$'),
        
        # Calculus and analysis
        ('∞', r'$\infty$'),
        ('∂', r'$\partial$'),
        ('∇', r'$\nabla$'),
        ('∫', r'$\int$'),
        ('∑', r'$\sum$'),
        ('∏', r'$\prod$'),
        ('√', r'$\sqrt{}$'),
        ('∘', r'$\circ$'),
        ('·', r'$\cdot$'),
        ('×', r'$\times$'),
        ('÷', r'$\div$'),
        ('±', r'$\pm$'),
        ('∓', r'$\mp$'),
        
        # Special characters
        ('−', '-'),  # Unicode minus to ASCII minus
        ('–', '--'),  # En dash
        ('—', '---'),  # Em dash
        ('′', "'"),  # Prime
        ('″', "''"),  # Double prime
        ('…', r'\ldots'),
        ('⟨', r'$\langle$'),
        ('⟩', r'$\rangle$'),
        ('⌊', r'$\lfloor$'),
        ('⌋', r'$\rfloor$'),
        ('⌈', r'$\lceil$'),
        ('⌉', r'$\rceil$'),
        ('|', r'$|$'),
        ('∥', r'$\|$'),
        ('∝', r'$\propto$'),
        ('↾', r'$\upharpoonright$'),
        ('↿', r'$\upharpoonleft$'),
        ('⇀', r'$\rightharpoonup$'),
        ('⇁', r'$\rightharpoondown$'),
        ('⇂', r'$\downharpoonright$'),
        ('⇃', r'$\downharpoonleft$'),
        ('↼', r'$\leftharpoonup$'),
        ('↽', r'$\leftharpoondown$'),
        ('↾', r'$\restriction$'),  # restriction
        ('⊥', r'$\bot$'),
        ('⊤', r'$\top$'),
        ('★', r'$\star$'),
        ('†', r'$\dagger$'),
        ('‡', r'$\ddagger$'),
        
        # Subscripts and superscripts (common patterns)
        ('₀', r'$_0$'),
        ('₁', r'$_1$'),
        ('₂', r'$_2$'),
        ('₃', r'$_3$'),
        ('₄', r'$_4$'),
        ('₅', r'$_5$'),
        ('₆', r'$_6$'),
        ('₇', r'$_7$'),
        ('₈', r'$_8$'),
        ('₉', r'$_9$'),
        ('⁰', r'$^0$'),
        ('¹', r'$^1$'),
        ('²', r'$^2$'),
        ('³', r'$^3$'),
        ('⁴', r'$^4$'),
        ('⁵', r'$^5$'),
        ('⁶', r'$^6$'),
        ('⁷', r'$^7$'),
        ('⁸', r'$^8$'),
        ('⁹', r'$^9$'),
        ('ⁿ', r'$^n$'),
        
        # Category theory
        ('◦', r'$\circ$'),
        ('⊣', r'$\dashv$'),
        ('⊢', r'$\vdash$'),
        
        # Increment/delta
        ('∆', r'$\Delta$'),
        ('∇', r'$\nabla$'),
        
        # Misc
        ('ℕ', r'$\mathbb{N}$'),
        ('ℤ', r'$\mathbb{Z}$'),
        ('ℚ', r'$\mathbb{Q}$'),
        ('ℝ', r'$\mathbb{R}$'),
        ('ℂ', r'$\mathbb{C}$'),
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
    
    return text

def main():
    with open('TKS_v7_4_FULL_BEAUTIFUL.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixed = fix_unicode(content)
    
    with open('TKS_v7_4_FULL_BEAUTIFUL.tex', 'w', encoding='utf-8') as f:
        f.write(fixed)
    
    print("Unicode characters fixed")

if __name__ == '__main__':
    main()
