# AI Explanation Format Standard

## Purpose

This file defines a standard format for explaining technical concepts to the user. **Any AI assistant working in this codebase should use this format when the user asks for explanations.**

---

## When To Use This Format

Automatically use this format when the user's question contains phrases like:

- "explain this to me"
- "what does ___ mean"
- "define these for me"
- "break this down"
- "in simple terms"
- "like I'm a beginner"
- "with metaphors"
- "6th grade" / "5th grade" / "simple vocabulary"
- "ELI5" (explain like I'm 5)
- "help me understand"
- "what is ___"
- "how does ___ work"
- "dumb it down"

**If the question is asking for understanding (not code), use this format.**

---

## The Format Structure

### For Each Concept, Provide:

#### 1. Three-Column Table Header

```markdown
| Metaphor | Technical | 6th Grade |
|----------|-----------|-----------|
| **[Relatable comparison]** | [Actual technical term/definition] | [Simple vocabulary version] |
```

#### 2. "Like This" Analogy

Start with **"Like this:"** and give a concrete, everyday example a child would understand.

#### 3. Bullet Points with Numbers

- Show the range (bad → good)
- Show current state
- Show goal or warning signs

#### 4. Use Horizontal Rules

Separate each concept with `---` for visual clarity.

---

## Complete Example

Here is a complete example explaining "Learning Rate":

---

### Learning Rate

| Metaphor | Technical | 6th Grade |
|----------|-----------|-----------|
| **Step size when walking** | Gradient descent step multiplier | How big of jumps the AI takes when learning |

**Like this:** Imagine you're walking to find the lowest spot in a hilly field while blindfolded.
- **Big steps (high LR)** = You move fast but might jump over the lowest spot
- **Tiny steps (low LR)** = You'll find it exactly but it takes forever
- **Just right** = Fast enough but still accurate

- Too high: > 0.01 (overshooting, unstable)
- Good range: 0.0001 to 0.001
- Too low: < 0.00001 (takes forever to learn)

---

## Quick Reference Card (Always End With This)

After explaining all concepts, provide a summary table:

```markdown
### Quick Reference Card

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| **Concept 1** | What's good | What's bad |
| **Concept 2** | What's good | What's bad |
```

---

## Style Rules

1. **No jargon without explanation** - Every technical term gets a simple version
2. **Concrete over abstract** - Use objects kids know (crayons, bikes, video games, report cards)
3. **Numbers with context** - Don't just say "0.5", say "0.5 (which means half)"
4. **Emojis optional** - Only if user's style includes them
5. **Keep metaphors consistent** - If you start with "video game", stick with gaming analogies
6. **Bold key terms** - Use **bold** for important words
7. **Short sentences** - Maximum 15 words per sentence in the "6th Grade" column

---

## Metaphor Bank (Common Concepts)

Use these tried-and-tested metaphors:

| Technical Concept | Recommended Metaphor |
|-------------------|---------------------|
| Loss function | Report card / Score on a test |
| Learning rate | Step size when walking blindfolded |
| Epochs | Re-reading a book multiple times |
| Batch size | Flashcards studied at once |
| Overfitting | Memorizing answers vs understanding |
| Underfitting | Not studying enough |
| Gradient | Slope of a hill |
| Weights | Importance knobs |
| Neurons | Brain cells / Team members |
| Layers | Floors in a building |
| Activation function | On/off switch with dimmer |
| Dropout | Randomly benching players during practice |
| Regularization | Rules to prevent cheating |
| Embedding | Secret code for words |
| Attention | Highlighting important parts |
| Transformer | Team that talks to each other |
| Encoder | Translator (English → code) |
| Decoder | Translator (code → English) |
| Latent space | Compressed imagination space |
| Checkpoint | Video game save file |
| Inference | Using what you learned on a test |
| Training | Studying before the test |
| Validation | Practice test |
| Temperature | Training wheels / Confidence dial |
| Entropy | Using all your crayons |
| Routing | Assigning group project tasks |
| Collapse | One person doing all the work |

---

## File Location

This file is located at:
```
docs/EXPLAIN_FORMAT_STANDARD.md
```

**AI Instruction:** When asked to explain technical concepts, check if `docs/EXPLAIN_FORMAT_STANDARD.md` exists and follow its format. If unsure whether to use this format, use it anyway - clear explanations are always better.

---

## Example Trigger Detection

**User asks:** "Can you define these metrics for me with metaphors?"

**AI should:**
1. Recognize trigger phrases: "define", "for me", "metaphors"
2. Load this format standard
3. Apply the three-column table + "Like this" + bullets structure
4. End with Quick Reference Card

**User asks:** "What's the loss function?"

**AI should:**
1. Recognize this is an understanding question, not a code question
2. Use this format even without explicit trigger words
3. Explain with metaphor, technical term, and simple vocabulary

---

## Version

- **Created:** 2026-01-07
- **Author:** Claude (based on user feedback)
- **Format Version:** 1.0
