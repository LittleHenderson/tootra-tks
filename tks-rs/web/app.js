// TKS Playground - Web Frontend
// Powered by WebAssembly

import init, { execute_tks, highlight_tks, get_examples, get_keywords } from './pkg/tkswasm.js';

// Note: The WASM module path is relative to index.html

class TKSPlayground {
    constructor() {
        this.editor = document.getElementById('editor');
        this.output = document.getElementById('output');
        this.highlightLayer = document.getElementById('highlight-layer');
        this.lineNumbers = document.getElementById('line-numbers');
        this.runBtn = document.getElementById('run-btn');
        this.shareBtn = document.getElementById('share-btn');
        this.examplesSelect = document.getElementById('examples');
        this.execTime = document.getElementById('exec-time');

        this.examples = [];
        this.keywords = [];
        this.isReady = false;
    }

    async init() {
        try {
            // Initialize WASM module
            await init();

            // Load examples and keywords
            this.examples = get_examples();
            this.keywords = get_keywords();

            // Populate examples dropdown
            this.populateExamples();

            // Bind event handlers
            this.bindEvents();

            // Initial render
            this.updateLineNumbers();
            this.updateHighlighting();

            // Check URL for shared code
            this.loadFromURL();

            this.isReady = true;
            this.output.textContent = 'Ready. Press Ctrl+Enter or click Run to execute.';

        } catch (err) {
            console.error('Failed to initialize TKS:', err);
            this.output.textContent = `Failed to load TKS engine: ${err.message}`;
            this.output.classList.add('error');
        }
    }

    populateExamples() {
        this.examples.forEach((example, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `${example.name} - ${example.description}`;
            this.examplesSelect.appendChild(option);
        });
    }

    bindEvents() {
        // Editor input
        this.editor.addEventListener('input', () => {
            this.updateLineNumbers();
            this.updateHighlighting();
        });

        // Sync scroll between editor and highlight layer
        this.editor.addEventListener('scroll', () => {
            this.highlightLayer.scrollTop = this.editor.scrollTop;
            this.highlightLayer.scrollLeft = this.editor.scrollLeft;
            this.lineNumbers.scrollTop = this.editor.scrollTop;
        });

        // Tab key handling
        this.editor.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = this.editor.selectionStart;
                const end = this.editor.selectionEnd;
                const value = this.editor.value;
                this.editor.value = value.substring(0, start) + '    ' + value.substring(end);
                this.editor.selectionStart = this.editor.selectionEnd = start + 4;
                this.updateHighlighting();
            }

            // Ctrl+Enter to run
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.run();
            }
        });

        // Run button
        this.runBtn.addEventListener('click', () => this.run());

        // Share button
        this.shareBtn.addEventListener('click', () => this.share());

        // Examples dropdown
        this.examplesSelect.addEventListener('change', (e) => {
            if (e.target.value !== '') {
                const example = this.examples[parseInt(e.target.value)];
                this.editor.value = example.code;
                this.updateLineNumbers();
                this.updateHighlighting();
                e.target.value = '';  // Reset dropdown
            }
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            this.updateLineNumbers();
        });
    }

    updateLineNumbers() {
        const lines = this.editor.value.split('\n');
        const lineCount = lines.length;

        let html = '';
        for (let i = 1; i <= lineCount; i++) {
            html += `<div>${i}</div>`;
        }
        this.lineNumbers.innerHTML = html;
    }

    updateHighlighting() {
        const source = this.editor.value;

        try {
            const result = highlight_tks(source);
            const tokens = result.tokens;

            // Build highlighted HTML
            let html = '';
            let lastEnd = 0;

            for (const token of tokens) {
                // Add any text before this token
                if (token.start > lastEnd) {
                    html += this.escapeHtml(source.substring(lastEnd, token.start));
                }

                // Add the token with its class
                const tokenText = source.substring(token.start, token.end);
                html += `<span class="tok-${token.class}">${this.escapeHtml(tokenText)}</span>`;

                lastEnd = token.end;
            }

            // Add any remaining text
            if (lastEnd < source.length) {
                html += this.escapeHtml(source.substring(lastEnd));
            }

            // Ensure there's always a trailing newline for proper scrolling
            if (!html.endsWith('\n')) {
                html += '\n';
            }

            this.highlightLayer.innerHTML = html;

        } catch (err) {
            // Fallback to plain text on error
            this.highlightLayer.textContent = source + '\n';
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    run() {
        if (!this.isReady) {
            this.output.textContent = 'TKS engine is still loading...';
            return;
        }

        const source = this.editor.value;

        if (!source.trim()) {
            this.output.textContent = 'No code to execute.';
            this.output.className = '';
            this.execTime.textContent = '';
            return;
        }

        // Show loading state
        this.output.textContent = 'Executing...';
        this.output.className = '';
        document.body.classList.add('loading');

        // Use setTimeout to allow UI update before execution
        setTimeout(() => {
            try {
                const result = execute_tks(source);

                document.body.classList.remove('loading');

                if (result.success) {
                    this.output.textContent = result.output || '()';
                    this.output.className = 'success';
                } else {
                    this.output.textContent = result.error || 'Unknown error';
                    this.output.className = 'error';
                }

                this.execTime.textContent = `${result.execution_time_ms.toFixed(2)}ms`;

            } catch (err) {
                document.body.classList.remove('loading');
                this.output.textContent = `Execution failed: ${err.message}`;
                this.output.className = 'error';
                this.execTime.textContent = '';
            }
        }, 10);
    }

    share() {
        const source = this.editor.value;
        const encoded = encodeURIComponent(source);
        const url = `${window.location.origin}${window.location.pathname}?code=${encoded}`;

        // Copy to clipboard
        navigator.clipboard.writeText(url).then(() => {
            const originalText = this.shareBtn.textContent;
            this.shareBtn.textContent = 'Copied!';
            setTimeout(() => {
                this.shareBtn.textContent = originalText;
            }, 2000);
        }).catch(() => {
            // Fallback: show URL in prompt
            prompt('Share URL:', url);
        });
    }

    loadFromURL() {
        const params = new URLSearchParams(window.location.search);
        const code = params.get('code');

        if (code) {
            this.editor.value = decodeURIComponent(code);
            this.updateLineNumbers();
            this.updateHighlighting();
        }
    }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    const playground = new TKSPlayground();
    playground.init();
});
