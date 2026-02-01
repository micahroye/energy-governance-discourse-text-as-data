# EU Energy Transition Discourse (Text-as-Data)

I built a mixed-methods measurement pipeline to detect **pro-renewables / pro-green framing** in EU energy discourse and track how rhetoric changes around major geopolitical and climate shocks.

## Why this matters
Environmental and energy policy outcomes are shaped not just by institutions, but by **how leaders frame tradeoffs** (security, affordability, competitiveness vs. decarbonization). This project operationalizes “pro-renewables framing” into a reproducible measure with transparent assumptions and evaluation.

## Core question
How does pro-renewables framing in EU energy discourse vary over time, especially around major shocks (e.g., Fukushima, Ukraine gas cutoff, Paris Agreement)?

## Methods
1. **Corpus construction:** extract and clean energy-related text; split into ~200-word chunks.
2. **Human labeling:** codebook applied to a stratified sample; disagreements resolved → **gold labels**.
3. **Supervised model:** multinomial **Naive Bayes** trained on gold labels; evaluated via cross-validation.
4. **LLM labeling:** structured prompt used to apply the same coding rules to additional text at lower marginal cost; comparisons run against gold labels.

## Model evaluation
- Naive Bayes performs well overall but is **conservative** on the positive class (precision > recall).
- LLM labeling can help scale classification while requiring careful calibration and validation against gold labels.
(See the memo for full metrics and discussion.)

## Outputs
- **Project memo (PDF):** [POLI176FinalMemo.pdf](POLI176FinalMemo.pdf)

### Requirements
- R (≥ 4.3 recommended)
- Packages: tidyverse, ggplot2, quanteda (plus any additional packages called in scripts)

### Run order
1. `src/NewCorpus.R` — build/clean corpus and chunk text
2. `src/newhandcode.R` — prep sample for manual coding (hand-coding workflow)
3. `src/coded.R` — (labeled hand coded data)
4. `src/goldlabel.R` — construct gold labels / resolve labels
5. `src/NewSupervised.R` — train/evaluate supervised model (Naive Bayes)
6. `src/LLM2.0.R` — run LLM labeling for scaling (API-based; see notes)
7. `src/LLMonGold.R` — compare LLM labels to gold labels / validation pass
8. `src/Graph.R` — generate figures/tables for memo/portfolio outputs


## Data access (not included)
This repo does **not** include the full raw corpus or labeled data files due to source/usage constraints.

## Notes on responsible use
- LLM outputs are treated as an **assistive measurement strategy**, not ground truth.
- Interpretation is limited by labeling ambiguity, class imbalance, and corpus selection.

## Next steps
- Expand labeled sample and add additional coders
- Compare Naive Bayes to regularized logistic regression / SVM

**Author:** Micah Roye
