# Open CAI Results Report

Generated from local artifacts copied back from the H100 run on 2026-06-22.

## Executive Summary

The first full Open CAI run succeeded end to end: generated data, trained SFT and DPO adapters for balanced/protective/permissive constitutions, copied artifacts locally, and ran fast capability, refusal, drift, constitution, and China-censorship evals.

The run validates the pipeline. DPO improved judged constitution faithfulness for all three adapters, and fast capability checks did not show a broad collapse.

The main tuning target is DPO refusal calibration. SFT mostly preserved the Heretic base refusal posture; DPO caused the large jump. The next pass should tune DPO data mix/selection/strength so the models refuse clearly harmful requests while still answering safe, analytical, or constitution-permitted requests directly.

## Eval Scope

These results are from fast eval configurations intended for iteration speed.

| Eval area | Dataset / method | Sample size | Notes |
| --- | --- | ---: | --- |
| Capability | HellaSwag | 100 | `acc_norm` reported below |
| Capability | ARC Challenge | 100 | `acc_norm` reported below |
| Capability | GSM8K | 10 | strict exact match reported below |
| Harmful refusal | `mlabonne/harmful_behaviors` | 100 | heuristic refusal detection |
| Behavior drift | `mlabonne/harmless_alpaca` | 100 | KL divergence vs Heretic base |
| Constitution faithfulness | DeepSeek v3.2 judge over fixed prompts | 10 | score out of 5 |
| China censorship | `promptfoo/CCP-sensitive-prompts` | 200 | heuristic refusal detection |

## SFT vs DPO Refusal Shift

The SFT-only adapters were evaluated with `configs/eval/sft-refusal-censorship.yaml`. These runs use the same marker-based refusal detector as the final DPO evals, but were run as a separate diagnostic pass, so baseline numbers differ slightly from the earlier DPO comparison because generation was rerun.

| Model | Harmful refusal | China censorship refusal |
| --- | ---: | ---: |
| Heretic base, SFT diagnostic run | 9/100, 9% | 6/200, 3.0% |
| Official Qwen3.5 4B, SFT diagnostic run | 95/100, 95% | 156/200, 78.0% |
| Balanced SFT | 11/100, 11% | 15/200, 7.5% |
| Balanced DPO | 98/100, 98% | 49/200, 24.5% |
| Protective SFT | 16/100, 16% | 20/200, 10.0% |
| Protective DPO | 99/100, 99% | 149/200, 74.5% |
| Permissive SFT | 8/100, 8% | 13/200, 6.5% |
| Permissive DPO | 90/100, 90% | 64/200, 32.0% |

Interpretation: the SFT pass mostly preserved the Heretic base refusal posture. The final refusal behavior is driven primarily by DPO.

The harmful-behavior shift is large: balanced moves from 11% at SFT to 98% after DPO, protective from 16% to 99%, and permissive from 8% to 90%. Because this eval set is mostly direct operational harm, the higher DPO refusal rate is not automatically bad. It does show that DPO is applying strong pressure.

The China-sensitive shift is more diagnostic. SFT remains close to the Heretic base at 6.5-10%, while DPO moves balanced to 24.5%, permissive to 32.0%, and protective to 74.5%. Protective nearly matches official Qwen3.5 4B, which is appropriate only if the protective constitution is meant to inherit that level of restriction. Balanced and permissive remain much less censorious than official Qwen, but they clearly moved away from the base model.

This narrows the next tuning surface: adjust preference-pair selection, DPO mix, beta, learning rate, epochs, or DPO dataset balance before changing the SFT recipe.

## Model Comparison

| Model | Base model | Adapter | HellaSwag | ARC | GSM8K | Harmful refusal | China refusal | Constitution score | KL vs Heretic |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Heretic base | `tvall43/Qwen3.5-4B-heretic` | none | 0.68 | 0.54 | 0.70 | 9% | 3.5% | varies by guide | 0.00 |
| Official Qwen3.5 4B | `Qwen/Qwen3.5-4B` | none | 0.68 | 0.53 | 0.70 | 96% | 75.5% | 4.4 on balanced | n/a |
| Balanced DPO | `tvall43/Qwen3.5-4B-heretic` | `outputs/dpo/balanced` | 0.67 | 0.60 | 0.60 | 98% | 24.5% | 4.5 | 2.36 |
| Protective DPO | `tvall43/Qwen3.5-4B-heretic` | `outputs/dpo/protective` | 0.65 | 0.61 | 0.80 | 99% | 74.5% | 3.9 | 2.41 |
| Permissive DPO | `tvall43/Qwen3.5-4B-heretic` | `outputs/dpo/permissive` | 0.69 | 0.54 | 0.70 | 90% | 32.0% | 4.8 | 1.27 |

Notes:

- Constitution scores are guide-specific. The base model scored 4.1 against balanced, 3.6 against protective, and 4.1 against permissive.
- The official Qwen3.5 4B eval is labeled `qwen-original` in the raw artifacts and used the balanced constitution prompts only.
- GSM8K used only 10 examples, so differences should be treated as directional.

## Capability

| Model | HellaSwag `acc_norm` | ARC Challenge `acc_norm` | GSM8K strict exact match |
| --- | ---: | ---: | ---: |
| Heretic base | 0.68 | 0.54 | 0.70 |
| Official Qwen3.5 4B | 0.68 | 0.53 | 0.70 |
| Balanced DPO | 0.67 | 0.60 | 0.60 |
| Protective DPO | 0.65 | 0.61 | 0.80 |
| Permissive DPO | 0.69 | 0.54 | 0.70 |

Fast-pass capability looks broadly intact. HellaSwag is flat to slightly down, ARC improves for balanced/protective, and GSM8K is noisy because of the 10-example limit.

The useful takeaway is conservative: none of the adapters obviously broke basic benchmark behavior in this fast pass. The sample sizes are too small to claim capability gains, especially for GSM8K. For the next serious model-selection pass, increase benchmark limits before making quality claims.

## Refusal And Censorship

| Model | Harmful refusal | China censorship refusal |
| --- | ---: | ---: |
| Heretic base | 9/100, 9% | 7/200, 3.5% |
| Official Qwen3.5 4B | 96/100, 96% | 151/200, 75.5% |
| Balanced DPO | 98/100, 98% | 49/200, 24.5% |
| Protective DPO | 99/100, 99% | 149/200, 74.5% |
| Permissive DPO | 90/100, 90% | 64/200, 32.0% |

This is the clearest behavioral tradeoff in the run. The DPO models moved strongly toward refusal on the harmful-behavior set, where many refusals are appropriate because the prompts ask for direct operational harm. Protective also moved toward official Qwen behavior on China-sensitive prompts. Balanced and permissive are less censorious than official Qwen but still far above the Heretic base, so the qualitative question is whether those refusals are constitution-faithful or inherited censorship behavior.

A manual spot check supports splitting this into two questions:

- On harmful prompts, the Heretic base often answered requests for malware, fraud, drug trafficking, weapon construction, and other operational abuse. DPO refusals are usually directionally correct there.
- On China-sensitive prompts, several refusals include sovereignty or institutional boilerplate. That is the riskier signal: those responses may reflect inherited model bias rather than the intended constitution.
- For balanced and permissive, the target is not lower refusal in general. The target is better discrimination: refuse direct abuse, answer legitimate political discussion or analysis, and avoid unsupported state-aligned framing.

## Constitution Faithfulness

| Eval guide | Heretic base | DPO adapter | Delta |
| --- | ---: | ---: | ---: |
| Balanced | 4.1 | 4.5 | +0.4 |
| Protective | 3.6 | 3.9 | +0.3 |
| Permissive | 4.1 | 4.8 | +0.7 |

The DPO pass improved judge scores for every constitution. Permissive showed the largest improvement and highest absolute score in this fast pass.

This result is encouraging but should be treated as directional. The judge set has 10 prompts per guide, so it is good for fast iteration but not enough to establish robust constitutional behavior. The next eval pass should add more constitution-specific prompts, especially mixed-intent and safe-but-sensitive prompts where refusal calibration matters.

## Drift

| Model | KL divergence vs Heretic base | Prompts |
| --- | ---: | ---: |
| Heretic base | 0.00 | 100 |
| Balanced DPO | 2.36 | 100 |
| Protective DPO | 2.41 | 100 |
| Permissive DPO | 1.27 | 100 |

Permissive drifted the least from the base model. Balanced and protective moved more substantially, which matches the much higher refusal rates.

The drift numbers line up with the refusal results: protective and balanced moved further away from the Heretic base, while permissive stayed closer. This is useful for comparing constitutions, but it does not say whether drift is good or bad by itself. The right reading is behavioral: drift should be justified by better constitution adherence, not just movement away from the base model.

## Training Runs

All final adapters were trained from `tvall43/Qwen3.5-4B-heretic` on the H100 pod with one SFT pass followed by one DPO pass.

| Constitution | Stage | Trained rows | Dropped overlength rows | Runtime | Train loss | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Balanced | SFT | 68,223 | 377 | 3h 55m | 0.6489 | `outputs/sft/balanced` |
| Balanced | DPO | 13,279 | 18 | 56m | 0.3321 | `outputs/dpo/balanced` |
| Protective | SFT | 68,223 | 377 | 4h 00m | 0.6488 | `outputs/sft/protective` |
| Protective | DPO | 14,871 | 20 | 1h 02m | 0.3203 | `outputs/dpo/protective` |
| Permissive | SFT | 68,223 | 377 | 4h 00m | 0.6483 | `outputs/sft/permissive` |
| Permissive | DPO | 7,090 | 10 | 31m | 0.3400 | `outputs/dpo/permissive` |

## Data And Artifacts

| Artifact | Local path | Size / rows |
| --- | --- | ---: |
| Balanced generated data | `data/generated/balanced-observation-v1-full.jsonl` | 42,520 rows |
| Protective generated data | `data/generated/protective-observation-v1-full.jsonl` | 42,520 rows |
| Permissive generated data | `data/generated/permissive-observation-v1-full.jsonl` | 42,520 rows |
| Generated data directory | `data/generated` | 4.2 GB |
| Final SFT adapters | `outputs/sft/{balanced,protective,permissive}` | ~60 MB each |
| Final DPO adapters | `outputs/dpo/{balanced,protective,permissive}` | ~60 MB each |
| Eval outputs | `outputs/eval` | 3.8 MB |
| H100 logs | `logs` | 15 MB |
| Preserved pod state | `pod-state` | 764 KB |

Local cleanup note: the final adapters and eval artifacts are present locally. The prepared `data/training/protective/...` and `data/training/permissive/...` split files are not currently present locally, but they can be regenerated from the full generated JSONL files.

## Source Artifacts

Primary result files:

- `outputs/eval/balanced-fast/summary.json`
- `outputs/eval/protective-fast/summary.json`
- `outputs/eval/permissive-fast/summary.json`
- `outputs/eval/qwen-original-balanced-fast/summary.json`
- `outputs/eval/china-censorship/summary.json`
- `outputs/eval/sft-refusal-censorship/summary.json`

Primary training logs:

- `logs/sft-balanced-h100-20260621-212229.log`
- `logs/dpo-balanced-h100-20260622-033324.log`
- `logs/nightly-20260622-050542-sft-protective.log`
- `logs/nightly-20260622-050542-dpo-protective.log`
- `logs/nightly-20260622-050542-sft-permissive.log`
- `logs/nightly-20260622-050542-dpo-permissive.log`
