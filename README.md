# RESUS
Codes for our paper published in TOIS'22:
[RESUS: Warm-Up Cold Users via Meta-Learning Residual User Preferences in CTR Prediction](https://dl.acm.org/doi/10.1145/3564283)

This repo includes an example for training RESUS upon a DeepFM model on the MovieLens-1M dataset. The dataset is preprocessed and splitted already.

Requirements: Python 3 and PyTorch.

Please run `pretrain.py` to train a DeepFM first. Then run `main_nn.py` or `main_rr.py` to train our RESUS. The default value of `overfit_patience` is 2 with less time cost, but a higher one may produce better results.  

