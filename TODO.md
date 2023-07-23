- [x] implement baselines FAs with importance score logged
- [x] implement metric Suff & Comp and with top 20%
- [x] implement metric Soft Suff & Soft Comp

- [x] evaluate analogy Suff & Comp




 =====> CASS: 
- Code
- [x] rationale length set: [0.1, 0.2, 0.3]
  - can be done in bash/job script
  - ```sh
    #!/bin/bash

    rationalRatioSet=(
        0.05
        0.1
        0.2
        0.3
    )

    for rationalRatio in "${rationalRatioSet[@]}"; do
        echo "Run with rationalRatio: $rationalRatio"
        # run the program
        python evaluate_analogies.py --rational_size_ratio $rationalRatio --eva_output_dir "<specify dir>"
    done

    ```
  - [ ] Alternaive: Once batch is working, we can implement a more efficient version
- [x] need to be able define the token number of each sample for evaluate sufficiency and comprehensiveness, at the moment, we define a ratio
  1. run rationalization_results from `https://github.com/keyonvafa/sequential-rationales#gpt-2`
  2. use `src/evaluation/gen_map_rational_size.py` to generate a rational length mapping file from greedy results (specify by `--input-dir`)
  3. run `src/evaluation/evaluate_analogies.py` with parameter `--rational_size_file` to specify the output mapping file
- [x] for hard rationales only: paper rationales for sequential predictions, table 1, metrics, [Ratio, Ante and No D]
  - src/evaluation/evaluate_analogies-old.py
- [x] greedy search
  - [x] migration code has been restored
    - src/rationalization/migrate_results_analogies.py
  - [x] TODO: importance score will missing, consider to generate a pseudo importance score

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


0720

please noted, i have changed the folder for store data for different models, different models will predict differently and have different data analogies.

Q: 

  - [x] aggregate_rationalizer.py is a testing file? or we use it in the real process?
    - A: It provides class `AggregateRationalizer` for `run_analogies.py`
