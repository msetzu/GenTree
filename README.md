# GenTree

Decision trees are among the most popular supervised models due to their interpretability and knowledge representation resembling human reasoning. Commonly-used decision tree induction algorithms are based on greedy top-down strategies. Although these approaches are known to be an efficient heuristic, the resulting trees are only locally optimal and tend to have overly complex structures. On the other hand, optimal decision tree algorithms attempt to create an entire decision tree at once to achieve global optimality. We place our proposal between these approaches by designing a generative model for decision trees. Our method named GenTree first learns a latent decision tree space through a variational architecture using pre-trained decision tree models. Then, it adopts a genetic procedure to explore such latent space to find a compact decision tree with good predictive performance. We compare GenTree against classical tree induction methods, optimal approaches, and ensemble models. The results show that GenTree can generate accurate and shallow, i.e., interpretable, decision trees.

This repository contains the source code of GenTree and the datasets used in the experiments presented in the paper "Generative Model for Decision Trees".
This code has been used to carry on the experiemnts of the paper

> Guidotti, R., Monreale, A., Setzu, M., & Volpi, G. (2024). Generative Model for Decision Trees. AAAI 2024.

## PyTorch version

The original TensorFlow code is still in the repository root (`vae.py`, `gentree.py`, `experiments/`, ...).
A PyTorch port, with the same API and the same experiment scripts, is in [`gentree_torch/`](gentree_torch/):

```
gentree_torch/
  vae.py                    # CVAE in PyTorch (+ make_loader for DataLoaders)
  genetic_algorithm.py      # DEAP genetic search over the VAE latent space (batched decoding)
  gentree.py                # GenTree(dataset, device=None).run(...)
  tree_encoder.py, utils.py, vae_dataset_generator.py
  experiments/              # run_gentree, test_gentree, test_gentree_inc_depth, hyperparameter_tuning, info_datasets, latent_space.ipynb
  notebooks/                # PyTorch versions of the notebooks
```

```bash
pip install -r gentree_torch/requirements.txt
python gentree_torch/experiments/test_gentree.py        # results/ written to the current directory
```

```python
from gentree_torch import GenTree, fitness_a, load_data, set_seed

set_seed(42)
X_train, X_val, X_test, y_train, y_val, y_test = load_data("iris")
gt = GenTree("iris").run(X=[X_train, X_val], y=[y_train, y_val], depth=4,
                         fitness=fitness_a, vae_pop_size=(10000, 5000, 32))
gt.predict(X_test)
```

The PyTorch version fixes several bugs of the original code (for example, `omega` was ignored and the custom mutation did nothing),
so its results may differ from the TensorFlow version. See [`bugs.md`](bugs.md).
