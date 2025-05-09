# CausalInferenceRecSys

## Project Structure


* **`coat/`**
  Folder containing the original COAT dataset.

* **`coat_sg/`**
  Possibly a single-group variant or preprocessed version of COAT data.

* **`ml-100k/`**
  MovieLens 100k dataset directory.

* **`check_point_coat/`, `check_point_100k/`**
  Directories storing model checkpoints for the COAT and MovieLens-100k datasets, respectively.

* **`dataloader/`**
  Custom dataloaders for processing COAT and MovieLens datasets.

* **`coatOg.py`**
  Script for training and evaluating the original (OG) deconfounded matrix factorization model on COAT.

* **`trainOg.py`**
  General training script for deconfounded MF across different datasets.

* **`ogModel.py`**
  Implementation of the original deconfounded MF model, including VPF (Variational Poisson Factorization).

* **`coatVae.py`**
  Training script for the VAE-based deconfounded recommendation model on COAT.

* **`trainVae.py`**
  General training script for VAE-based deconfounded MF.

* **`vaeModel.py`**
  Implementation of the VAE model for deconfounded recommendation.

* **`coatNeu.py`**
  Training script for NeuMF (Neural Matrix Factorization) on COAT.

* **`trainNeu.py`**
  General training script for NeuMF-based deconfounded recommendation.

* **`neuModel.py`**
  Implementation of the NeuMF model.

* **`__pycache__/`**
  Python cache directory for compiled bytecode.

