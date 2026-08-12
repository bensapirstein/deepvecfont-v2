# External review of `report/REPORT.md`

Received 2026-08-12, from Gemini, on the report as of commit `2ef1bde`. Kept in the
repo rather than only in the vault because `docs/review-response.md` answers it point by
point and a runbook that cites a document nobody can open is not a runbook.

Verbatim below. Where the review misquotes one of our own numbers the correction is in
`docs/review-response.md` §1, not here — the point of keeping this verbatim is that it
stays the thing that was actually said.

---

### 1. Summary

The project attempts to reproduce DeepVecFont-v2 and evaluates 21 single-factor architectural and training changes against it. The authors successfully diagnose a cross-rasterizer metric mismatch that accounts for a large portion of the Chinese reconstruction gap, and they establish a rigorous seed-noise floor (0.0097 for Chinese L1) before testing interventions. However, the primary proposed architectural change (E9: modifying the noise schedule) fails to clear the authors' own pre-established noise floor. Furthermore, the admission that the validation metric is functionally uncorrelated with the reported test metric undermines the reliability of the checkpoint selection process across the entire study.

### 2. Section-by-section notes

- **Section 3 (Reconstruction results):** The authors identify a 0.0027 L1 discrepancy in English evaluation caused by using 50 samples per glyph instead of the paper's 10. Because this is purely an inference-time parameter, it is trivial to re-evaluate the test set at 10 samples to produce a directly comparable number. Failing to run this cheap evaluation leaves an unforced error in the final tables.

- **Section 4.1 & 5.3 (Rule pre-commitment vs. E9 results):** The authors establish a clear seed-noise floor of 0.0097 L1 and 0.0315 s-IoU, explicitly stating: "Any change producing a smaller difference than this cannot be told apart from a lucky seed". E9 yields deltas of 0.0040 and 0.0271, respectively. Both fall _inside_ the noise floor. The authors then pivot to sign-agreement (6 out of 6 seeds/metrics) to justify E9 as a successful direction. This violates the explicit pre-commitment. A post-hoc probabilistic argument (the 1 in 64 chance) cannot be used to bypass a pre-registered noise floor; by the report's own definitions, E9 is indistinguishable from noise.

- **Section 5.1 (The comparison table):** The table lacks standard deviations or variance markers for the "three-seed means", which is a glaring omission for a report entirely focused on resolving signal from noise. Additionally, there is a typo in the metric headers: "Error (L1) 1" and "SSIM 1", which appear to be misformatted downward/upward arrows.

- **Section 6.3 (Checkpoint selection):** The authors explicitly acknowledge that the validation score used to select checkpoints ranks candidates at a Spearman correlation of 0.125 against the actual reported metric. While the honesty is appreciated, selecting checkpoints using a metric that is effectively uncorrelated with the test metric invalidates the fine-grained comparisons between all 26 candidates.

### 3. Critical issues

- **Claiming a result inside the noise floor:** E9 does not clear the 0.0097 noise floor. Promoting it as the "Improved model" in Table 5.1 is unsupported by the quantitative evidence. If a strict floor is established, a delta inside that floor is a null result.

- **Validation metric mismatch:** Because the validation checkpointing strategy is defective, the "best" checkpoints are drawn somewhat randomly with respect to true performance. The authors note that fixing this would "invalidate all 113 scored rows". Time constraints aside, evaluating on a mismatched checkpoint selection protocol severely compromises the scientific validity of the sweep.

- **Incomplete reproduction pipeline:** The authors identified a massive data augmentation mismatch (6x vs 10x for Chinese) but state they did not test it because it required a retrain they "did not have time for". A baseline reproduction is incomplete until the data pipelines match; this is a missing fundamental step, not a future limitation.

### 4. Minor issues

- Missing evaluation at a `best-of-10` sampling budget for English to match the paper exactly.
- Missing standard deviations in the text of Table 5.1.
- The architectural change (E9) is correctly mapped to a required category ("Change the noise schedule"), but framing it as an "architecture" change in the section headers is slightly inaccurate; it is a regularization/training-dynamics intervention.

### 5. Specific questions the author should be able to answer

1. If E9 does not clear your own pre-established seed-noise floor, on what epistemological grounds are you featuring it as the "Improved model" in Table 5.1?
2. Given that checkpoint selection was functionally random with respect to the rendered L1 metric, how can we trust that the 0.0101 spread of the 26 candidates isn't purely an artifact of mismatched checkpoint sampling?
3. Why was the English evaluation not simply re-run at 10 samples per glyph to definitively close the 0.0027 gap, given that inference is cheap and requires no retraining?

### 6. Overall assessment

**Not ready to submit.**

While the methodological rigor in isolating confounding variables—such as rasterizer differences and random initialization streams—is exceptionally high, the core empirical claims are fatally compromised. The report establishes an excellent framework for scientific skepticism but fails to apply that same skepticism to its own "winning" candidate.

**Single highest-priority fix:**

Re-evaluate the baseline and the top candidates (E9, and the degrading E1) using rendered-metric checkpoint selection. If E9 still fails to clear the noise floor after proper checkpointing, the authors must report the entire sweep as a true null result and remove E9 from the "Improved" column of Table 5.1, rather than attempting to salvage it through post-hoc sign agreement.
