# Experiment Explanations

Before every autoresearch benchmark run, write one file:

```text
explanations/run_###.md
```

Use zero-padded run numbers: `run_001.md`, `run_002.md`, etc. `autoresearch.sh` enforces this. If an explanation is missing, the run should not happen.

## Required template

```markdown
# Run ### — <short title>

## Proposed change
What will be changed and which files will be touched?

## Source and evidence
Where did the idea come from?

Use one or more:
- course material: <lecture/note/section>
- agent prior knowledge
- web search: <URL/title>
- paper: <paper title/authors>
- code search: <repo/example>
- experiment observation: <previous run numbers>

## Course-material connection
Explain the Information Theory connection: cross-entropy, redundancy, universal coding, Markov/source modeling, arithmetic-coding interpretation, etc.

## Hypothesis
Why should this lower bits/symbol or improve reliability/runtime?

## Risks
What could go wrong? Overfitting, sparse contexts, runtime, memory, invalid probabilities, presentation weakness, etc.

## Validation plan
Which command will be run? What metric change would count as success? What checks must pass?
```

## Reading rule

At the start of each run, read the last five `run_*.md` files to avoid repeating dead ends. The `autoresearch.hooks/before.sh` hook also surfaces a compact view of the last five explanations.
