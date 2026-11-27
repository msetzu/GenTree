# Bugs found in the original (TensorFlow) code

The original files are unchanged. "Fixed" means fixed in `gentree_torch/`.

## Affect results
| # | Where | Bug | Status |
|---|-------|-----|--------|
| 1 | `genetic_algorithm.py:167,169,319` | `omega` is stored but never passed to the fitness function, which always uses its default of 0.75. The `omega` sweep in `hyperparameter_tuning.py` therefore has no effect. | Fixed |
| 2 | `genetic_algorithm.py:174-178` | `__mutate` makes a new latent vector but throws it away; the individual is not changed. Custom mutation does nothing except force a re-evaluation. | Fixed (in-place update) |
| 3 | `experiments/run_gentree.py:81` | `fitness_value = gt.best_dt["n_nodes"]`, so the best tree over all runs is the one with the fewest nodes, not the lowest fitness. It should use `val_fitness_value`. | Fixed |
| 4 | `experiments/test_gentree_inc_depth.py:23,74` | The header has no `tot_time` columns but the rows do, so every time column is shifted by two. `pickle.dump(result[19])` saves `mean_gp_time` instead of the best tree (`result[21]`). | Fixed |
| 5 | `genetic_algorithm.py:131` | `initial_tree` is encoded without the MinMax scalers before `model.encode`. | Fixed |
| 6 | `vae.py:93` | The default `Adam` optimizer is created once, when the function is defined, so all `CVAE` instances in a process share it and its state. | Fixed (new optimizer per `fit`) |
| 7 | `experiments/*.py` (`np.random.seed(42)`) | Seeding has no effect: the code uses Python `random` and TF RNGs, and trees are generated in joblib workers. | Fixed (`set_seed` plus per-tree seeds passed to the workers) |

## Crashes / compatibility
| # | Where | Bug | Status |
|---|-------|-----|--------|
| 8 | `tree_encoder.py:63`, `utils.py:61` | The sklearn node dtype is hard-coded (7 fields). sklearn ≥1.3 adds `missing_go_to_left`, so `Tree.__setstate__` fails. | Fixed (uses `state['nodes'].dtype`) |
| 9 | `vae_dataset_generator.py:68` (and RF variants) | A test tree deeper than every training tree crashes `encode` (IndexError). | Fixed (such trees are skipped) |
| 10 | `vae_dataset_generator.py:89-96` | `randint(100, n_samples)` raises when the training set has fewer than 100 rows (`NEW_RF`). | Fixed |
| 11 | `prepare_datasets.py:10` | `OneHotEncoder(sparse=False)` was removed in sklearn 1.4 (now `sparse_output`). | Not fixed (file not ported) |
| 12 | `prepare_datasets.py:1621…` | Paths are `../datasets/` but the script is in the repo root. Shuffles use no `random_state`. | Not fixed |
| 13 | `experiments/hyperparameter_tuning.py:65` | `nbr_training_trees()` has no `path` parameter and reads a global `path`, so it only works under `__main__`. | Fixed |
| 14 | `experiments/*.py` | The `results/...` folders are never created, so `open()` fails on a fresh clone. | Fixed |

## Minor
| # | Where | Bug | Status |
|---|-------|-----|--------|
| 15 | `tree_encoder.py:164` | The feature range check allows `n_features + 1` (off by one). In practice it never triggers. | Fixed |
| 16 | `tree_encoder.py:196` | `complete_tree` uses `feature > 0`, which skips internal nodes that split on feature 0. It should be `>= 0`. | Fixed |
| 17 | `genetic_algorithm.py:125` | Compares `tree.shape[0]` (rows) with the input length. It should be `shape[1]`. Works only by accident. | Fixed |
| 18 | `vae.py:126,132` | Early stopping: `delta_count` is never reset (cumulative, not consecutive). The `elbo > 0 and elbo > previous` case counts an *improvement* as worse. That case is unreachable because ELBO < 0 here. | Kept as-is, to match the original |
| 19 | `experiments/latent_space.ipynb` | `generate_latent_trees` calls `fit_transform`, refitting the scalers on the trees being projected; it should use `transform`. The CSV header `"id" + "l0,"…` is missing a comma. | Fixed |
