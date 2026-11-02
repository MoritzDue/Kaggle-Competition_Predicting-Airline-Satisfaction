# Predicting Airline Satisfaction | Kaggle S6E10

**Final rank: 1,249 / 3,735 (Top 34%)** · **Private ROC AUC: 0.96177** · **Public ROC AUC: 0.96080**

[Competition](https://www.kaggle.com/competitions/playground-series-s6e10) · [Leaderboard](https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard)

## Project summary

Predicted airline passenger satisfaction from tabular data (699,635 training rows; 299,844 test rows). I compared **CatBoost, XGBoost, LightGBM, RealMLP, feature engineering, and weighted ensembles**, using five-fold stratified cross-validation and ROC AUC. My final entry combined **my feature-engineered LightGBM model** with **publicly shared TabPFN-3.5 predictions** through a logistic stacking model.

## What worked

**Leakage-aware feature engineering was the clearest improvement in my own models.** I investigated the repeated values of `Flight Distance`, creating count encoding and smoothed target encoding. Training target encodings were cross-fitted *within each outer training fold* so validation labels were not used to build features.

| LightGBM features | 5-fold OOF ROC AUC |
|---|---:|
| Original features | 0.958810 |
| + Flight Distance count encoding | 0.959526 |
| + Smoothed target encoding | 0.959974 |
| + Both features | **0.960196** |

The combined features improved ROC AUC by **+0.001386** over the original LightGBM.

## Model progression

| Approach | Private leaderboard ROC AUC |
|---|---:|
| CatBoost baseline | 0.95821 |
| RealMLP | 0.95864 |
| CatBoost + XGBoost + LightGBM ensemble | 0.95931 |
| Flight Distance engineered blend | 0.96039 |
| Improved LightGBM (67%) + RealMLP (33%) blend | 0.96085 |
| Public TabPFN predictions alone | 0.96161 |
| **LightGBM + public TabPFN logistic stack (final)** | **0.96177** |

**Takeaway:** More extensive blending of my tree and neural models did not outperform the public TabPFN predictions. Combining TabPFN with my improved LightGBM may have added a little complementary signal, although the final improvement was marginal (+0.00016 private AUC).

## Files

Selected notebooks from the experimentation process; intermediate experiments are omitted.

- [`01_eda.ipynb`](01_eda.ipynb) — exploratory analysis.
- [`04_cross_validation_ensemble.ipynb`](04_cross_validation_ensemble.ipynb) — tree models and OOF ensembles.
- [`08_flight_distance_experiments.ipynb`](08_flight_distance_experiments.ipynb) — feature-engineering ablation and test predictions.
- [`13_final_submission_lgb_tabpfn.ipynb`](13_final_submission_lgb_tabpfn.ipynb) — logistic stacking using saved prediction arrays.

## Acknowledgement

The final ensemble uses publicly shared TabPFN-3.5 predictions from the Kaggle community (see [Tabular Foundation Models discussion](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746363)).

*Leaderboard scores and final ranking are from Kaggle submission history. Cross-validation and private leaderboard scores evaluate different data samples.*
