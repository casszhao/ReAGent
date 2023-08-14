# WIP

For the analogies experiment, we use the [analogies dataset](https://aclweb.org/aclwiki/Google_analogy_test_set_(State_of_the_art)) provided by [Mikolev et al](https://arxiv.org/abs/1301.3781). The dataset is already included in our Github, so there is no need to download anything else.

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


### reproducing baseline greedy search and exhaustic search

we use the code from the origin paper, Rationales for Sequential Predictions, to run the greedy search (using their off shef compatible GPT2 model) and exhaustive search.

We then modify their output format to the same format as ours (`src/rationalization/migrate_results_analogies.py`). As they only provide rationales, so we fill the importance scores as 0 for non-rationales. For rationales, the importance scores are 1/n, n is the length of the rationales. It is a pesudo importance scores for using unified evaluation pipeline. 