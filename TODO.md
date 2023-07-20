- [x] implement baselines FAs with importance score logged
- [x] implement metric Suff & Comp and with top 20%
- [x] implement metric Soft Suff & Soft Comp

- [x] evaluate analogy Suff & Comp




 =====> CASS: 
- Code
- [ ] rationale length set: [0.1, 0.2, 0.3]
- [ ] need to be able define the token number for evaluate sufficiency and comprehensiveness, at the moment, we define a ratio
- [ ] for hard rationales only: paper rationales for sequential predictions, table 1, metrics, [Ratio, Ante and No D]
- [ ] greedy search

We will test on other model and feature attribution too. Do feature attribution first



I know we have done attention, but compare to previous paper, we need at least another 3. I have some suggestion, but due to the time limit, please do the ones which are fast. package inseq


- Code
  - [ ] FAs: integrated grads, 
  - [ ] FAs: Grad norms 
  - [ ] FAs: gradients * embeddings, 
  - [x] FAs: attention rollout (config.json - importance_score_evaluator.attention.type = rollout)
  - [x] FAs: last attention (config.json - importance_score_evaluator.attention.type = last)
  - [x] FAs: attention all (config.json - importance_score_evaluator.attention.type = all)
  - [ ] Alternative: intergrate module inseq
- Results
  - [ ] FAs: integrated grads, 
  - [ ] FAs: gradients, 
  - [ ] FAs: gradients * embeddings, 
  - [ ] FAs: attention rollout
  - [ ] FAs: last attention
  - [ ] FAs: attention all
  - [ ] Alternative: intergrate module inseq





- [ ] model: KoboldAI/OPT-6.7B-Erebus (too big to load locally, can r)
- [ ] model: openlm-research/open_llama_7b_v2 (downloading killed)
- [ ] model: bigscience/bloom (half hour downloading, only done 10%)



====> cass done
0719
1. have done extracting importance scores and evaluation for the 3 attention FA, using local_attention.sh (3 different config). runnable. 
2. accidently delete our methods extracted importance scores. SORRY!