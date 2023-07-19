- [x] implement baselines FAs with importance score logged
- [x] implement metric Suff & Comp and with top 20%
- [x] implement metric Soft Suff & Soft Comp

- [x] evaluate analogy Suff & Comp
- [ ] intergrate module inseq




 =====> CASS: 
 
- [ ] need to be able define the token number for evaluate sufficiency and comprehensiveness, at the moment, we define a ratio


We will test on other model and feature attribution too. Do feature attribution first



I know we have done attention, but compare to previous paper, we need at least another 3. I have some suggestion, but due to the time limit, please do the ones which are fast. package inseq
- [ ] FAs: integrated grads, 
- [ ] FAs: gradients, 
- [ ] FAs: gradients * embeddings, 
- [ ] FAs: attention rollout
- [ ] FAs: last attention
- [ ] FAs: 




- [ ] model: KoboldAI/OPT-6.7B-Erebus (too big to load locally)
- [ ] model: openlm-research/open_llama_7b_v2 (downloading killed)
- [ ] model: bigscience/bloom (half hour downloading, only done 10%)