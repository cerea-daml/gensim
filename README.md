# GenSIM – Generative Sea‑Ice Model

This is the official implementation of **GenSIM**, a generative sea-ice model to learn sea-ice dynamics with neural networks and flow matching.

## Repository Structure

``` 
data/
├─ auxiliary – Auxiliary data contained in the repository
├── ds_auxiliary.nc - Auxiliary data file (grid cells, mask, bathymetry)
├─ train_data – Zarr training data (not contained in the repository and has to be linked)

gensim/
├─ augmentation.py – data augmentation (flips, rotations, patch generation)
├─ data_module.py – LightningDataModule for training/validation datasets
├─ dataset.py – PyTorch Dataset that reads Zarr data
├─ embedding.py – random‑Fourier embeddings and Embedder
├─ encoder_decoder.py – Encoder and Decoder to map to physical space
├─ model.py – LightningModule handling training and forecasting logic
├─ network.py – Transformer architecture (tokenizer, attention, skips)
├─ sampler.py – Flow‑matching sampler with schedule and second‑order update
├─ utils.py – helper functions (masking, averaging, param grouping)
└─ wrapper.py – PatchedNetwork wrapper for forecasting with domain decomposition

config.yaml – Composed Hydra configuration of the training run for GenSIM
environment.yml – Conda environment definition
setup.py – Package installation script
train.py – Entry point for training (Hydra CLI)
LICENSE – MIT license
README.md – This file
```

## Installation

```bash
git clone https://github.com/cerea-daml/gensim.git
cd gensim
conda env create -f [`environment.yml`](environment.yml)
conda activate gensim
pip install -e .
```

Verify installation:

```bash
python -c "import gensim; print(gensim.__version__)"
```

## Configuration

The file [`config.yaml`](config.yaml) contains all hyper‑parameters. Key sections:

- `trainer` – Lightning trainer settings (accelerator, devices, precision, max_steps).
- `surrogate.network` – Transformer architecture (n_input, n_output, n_features, n_blocks, etc.).
- `surrogate.encoder` / `decoder` – Encoder/decoder parameters.
- `surrogate.sampler` – Number of steps and schedule parameters.
- `surrogate.train_augmentation` – Probabilities for flips/rotations.
- `data` – Paths to data, batch size, number of workers.

You can override any entry from the command line, e.g.:

```bash
python [`train.py`](train.py) trainer.max_steps=500000 data.batch_size=8 exp_name=my_exp
```

## Training

Ensure the `data/train_data` folder contains the required Zarr files (`train.zarr`, `validation.zarr`) and the auxiliary NetCDF (`auxiliary/ds_auxiliary.nc`).

```bash
python [`train.py`](train.py)
```

The script logs progress with a tqdm bar, saves checkpoints under `data/models/<exp_name>/`, and (offline) logs to Weights & Biases as configured in `config.yaml`.

To resume training set `ckpt_path` in `config.yaml` to the desired checkpoint.

## Data Layout

Sea‑ice state variables: `sit`, `sic`, `sid`, `siu`, `siv`, `snt`.  
Forcing variables: `tus`, `rhus`, `uas`, `vas`.  

The Zarr file contains a `datacube` array with dimensions `[time, variable, y, x]` and a `var_names` attribute.  

The auxiliary NetCDF provides `mask`, `bathymetry`, `x_coord`, `y_coord`.

## License

This project is released under the MIT License (see `LICENSE`).

## Citation

If you use GenSIM, please cite:

```bibtex
@article{Finn2025GenSIM,
  author = {Tobias Sebastian Finn},
  title = {Generative Sea‑Ice Modeling with Flow‑Matching Transformers},
  journal = {Journal of Climate Modeling},
  year = {2025},
  volume = {XX},
  pages = {YY--ZZ},
  doi = {10.1234/jcm.2025.xxx}
}
```

## Contact

Tobias Sebastian Finn – tobias.finn@enpc.fr

*End of README*