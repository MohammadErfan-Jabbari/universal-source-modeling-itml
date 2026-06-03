# Explanation Template

Before each autoresearch run, write a file `explanations/run_###.md` with the following sections.

## Run ### — [Short Title]

### Proposed change
What predictor are you adding and what does it do differently?

### Source and evidence
Which prior run, idea file, or course material motivated this?

### Course-material connection
Which information-theoretic concept justifies this approach?

### Hypothesis
Why do you expect this to improve bits/symbol?

### Risks
What could go wrong?

### Validation plan
```bash
PREDICTOR_PATH=submissions/your_predictor.py ./autoresearch.sh
```
