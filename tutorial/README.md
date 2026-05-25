# MASA tutorial notebooks

```bash
pip install -e . --group tutorials
```

Notebooks:

1. `01_minipacman_labelled_mdp_demo.ipynb` - labelled MiniPacman MDP, labels/costs, rendered rollout.
2. `02_colour_bomb_product_mdp_demo.ipynb` - ColourBombGridWorld DFA/product-state demo for the diffuse-bombs monitor.
3. `03_pacman_coins_safety_abstraction_demo.ipynb` - Pacman-with-coins concrete vs abstract safety states.
4. `04_probabilistic_shielding_training_comparison.ipynb` - plain PPO, model-free RECREG, and probabilistic-shielded ParameterizedPPO with TensorBoard curve extraction.

The notebooks intentionally include source-code inspection cells (`inspect.getsource`) so they continue to show the current MASA source snippets even after changes in the library.
