# Long-sequence ID-based video recommendation: architecture research plan

**Date:** 2026-11-19  
**Branch:** `experiment/sliding-window-history-vs-disjoint-1day`  
**Status:** Design / data audit required; no new model training or claims of superiority.  
**Frozen branch:** `feature/e2e-integration-contract-20261117` — never modify.

## Research question

Given an event-time 24-hour history capped at **2048** ID-based interactions, can a model retain quality close to a **matched Full-2048 backbone** while reducing measured **end-to-end FLOPs/latency/memory**, compared with **the best feasible fixed-history baseline**? Do not conflate max cap with every request having 2048 interactions. Do not treat Full as a theoretical upper bound on accuracy.

## Dataset reality check

Official KuaiRand-1K contains ~1,000 users, 4,369,953 items, 11,713,045 interactions, 30 user features, 62 item features, 12 feedback signals and timestamps. Files: standard/random logs, user_features_1k.csv, video_features_basic_1k.csv, video_features_statistic_1k.csv. Basic video features include author_id, tag, music_id/type, video_duration and upload information. Supplementary video categories and captions exist, with coverage to be verified. **Dense pretrained video-content embeddings are not guaranteed.** Current Phase7 pipeline is ID-centric; presence of feature files locally and join coverage must be audited. Sources: https://github.com/chongminggao/KuaiRand ; https://kuairand.com/ ; https://doi.org/10.1145/3511808.3557624 .

**Mandatory audit:** inventory raw files and columns, confirm raw/reindexed video_id mapping, coverage of metadata among candidate and historical videos, null/missing tag and caption fractions, timestamps and availability at request time, repeated IDs, user/session distribution, history-length quantiles, label/feedback availability, and split leakage. Video statistics can encode future engagement; exclude unless temporally reconstructed. Never leak target feedback into history. Treat random exposure and standard exposure separately in analyses.

## Architecture choices

1. **Strong sequence baselines:** existing mean pooling (historical reference); GRU4Rec-style recurrent encoder (O(Ld²) rough compute); SASRec causal self-attention (O(L²d + Ld²) compute, O(L²) attention weights). Train/evaluate Full-2048 and Fixed-50 using the **same backbone and training budget**. Report padding/masking, positional/time encoding and history selection semantics.
2. **Target-aware retrieval (first priority):** use learned ID embedding or metadata-derived representation; retrieve top-M interactions relevant to candidate. Exact 2048×d dot products already have cost O(Ld), so compare with simple vectorized exact scan before adding SimHash/LSH. LSH entails hash generation, index construction/updates, bucket lookup and recall loss; bitwise comparisons are not free end-to-end. Dynamic M in a bounded set e.g. {10,20,50,100}, with request-time features and quality constraint; ensure selection precedes expensive sequence encoder. Target-aware method is not a pure independent two-tower user encoder because user representation depends on candidate.
3. **Differentiable pruning:** lightweight scorer over all L items; straight-through/Gumbel gates and explicit budget penalty. Compare against deterministic top-k and random gates; monitor collapse and training/inference mismatch. Scoring all 2048 has nonzero cost; count it.
4. **Adaptive merging:** session/local similarity or duplicates grouped into meta-tokens, carrying count, recency/duration and position. Must not claim lossless semantic preservation; merging may erase order and rare-interest signals. Compare against deterministic dedup/session pooling. Compression cost and dense attention quadratic cost both count.
5. **Hybrid:** coarse recency window + low-cost target-aware retrieval from older events, dedup and optional dynamic gate; only if standalone ablations justify it.

## Evaluation correction

Current label is binary `long_view`, so **ROC-AUC and log-loss** are the primary directly supported metrics for the existing request-level pointwise prediction contract. **NDCG@10/HR@10 require well-defined per-user/time candidate sets with multiple candidates, valid relevance labels and appropriate exposure assumptions.** Do not fabricate rankings or treat a single observed impression as a ten-candidate recommendation list. Candidate-ranking experiment requires separate documented construction, leakage controls and ranking protocol; negative sampling changes the estimand.

Compare Full-2048, Fixed-50, Fixed-100, random-50, uniform-50, and adaptive methods on the **same model family**, splits, candidate pool, data access, device, and quality tolerance. Add metadata/no-metadata ablations; recency-only vs target-aware vs target-unaware compression; exact vs approximate retrieval; static vs adaptive budget; oracle upper bound only on development data, never deployable.

Cost accounting: history materialization and transfer, metadata/feature extraction, target embedding lookup, ANN/hash index build amortization and updates, retrieval and selection, encoder forward, policy overhead, CPU/GPU synchronization, peak RAM/VRAM, p50/p95 warmed per-request latency and throughput under stated batching. Distinguish offline training FLOPs, theoretical FLOPs and measured inference latency. Avoid counting a Full-history fetch as K-aware savings.

## Protocol and gates

- Frozen legacy E2E, Phase7, V4 and Shared Variable-K V1 stay untouched.
- Existing test has already been viewed; do not use it for architecture/policy selection or describe subsequent evaluations on it as pristine confirmatory results.
- Before model building, audit local raw feature availability and ID mapping.
- First train **matched** SASRec/GRU baselines with Full and Fixed-50, alongside the frozen mean-pooling reference. Use user-disjoint or time-aware leakage controls consistent with the declared estimand.
- Next run simple exact target-aware top-M retrieval vs recency-M under the same trained backbone. Optimize dynamic gating only if a measurable opportunity exists.
- Select all architectures/hyperparameters/penalties using development-only data; establish a genuinely independent confirmatory evaluation plan or explicitly mark results exploratory.
- Reject adaptive approaches dominated by feasible fixed policies in quality and total cost; report negative results.
