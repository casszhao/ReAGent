# WIP

## Repo structure

### Source code

- `src/data/prepare_evaluation_analogy.py`: dataset preparation of our rationalization approach on analogies
- `src/rationalization/random_replacing/run_analogies.py`: analogies experiment of our rationalization approach on analogies
- `src/rationalization/random_replacing/evaluate_analogies.py`: metric evaluation of our rationalization approach on analogies
- `src/rationalization/random_replacing/migrate_results_analogies.py`: file migration from others rationalization results to ours
- `src/rationalization/random_replacing/test_POSTagTokenSampler.py`: to be removed
- `src/rationalization/greedy_masking`: core of others rationalization approach
- `src/analogies.py`: run analogies with others rationalization approach (need to be re-organized)
- `src/analysis`: not in use
- `src/rationalization/greedy_replacing`: not in use
- `src/utils`: not in use
- `src/*.py` except `src/analogies.py`: not in use

### Others

- `config`: configuration of our rationalization approaches
- `data`: dataset for rationalization experiments
- `docs`: documentations of our approaches (for `src/rationalization/random_replacing/rationalizer`)
- `evaluation_results`: results of metric evaluation
- `jobs`: job scripts for HPC
- `logs`: logs
- `notes`: misc
- `rationalization_results`: results of rationalization
