# GenTree

Decision trees are among the most popular supervised models due to their interpretability and knowledge representation resembling human reasoning. Commonly-used decision tree induction algorithms are based on greedy top-down strategies. Although these approaches are known to be an efficient heuristic, the resulting trees are only locally optimal and tend to have overly complex structures. On the other hand, optimal decision tree algorithms attempt to create an entire decision tree at once to achieve global optimality. We place our proposal between these approaches by designing a generative model for decision trees. Our method named GenTree first learns a latent decision tree space through a variational architecture using pre-trained decision tree models. Then, it adopts a genetic procedure to explore such latent space to find a compact decision tree with good predictive performance. We compare GenTree against classical tree induction methods, optimal approaches, and ensemble models. The results show that GenTree can generate accurate and shallow, i.e., interpretable, decision trees.

This repository contains the source code of GenTree and the datasets used in the experiments presented in the paper "Generative Model for Decision Trees".
This code has been used to carry on the experiemnts of the paper

> Guidotti, R., Monreale, A., Setzu, M., & Volpi, G. (2024). Generative Model for Decision Trees. AAAI 2024.
