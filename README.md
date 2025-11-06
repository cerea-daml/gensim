# GenSIM – Generative Sea‑Ice Model

This is the official implementation of **GenSIM**, a generative sea-ice model to learn sea-ice dynamics with neural networks and flow matching.

[![Demo](https://img.shields.io/badge/Demo-Colab-F9AB00?style=flat&logo=googlecolab&color=%23F9AB00)](https://colab.research.google.com/drive/1R3KPE4okFUGRcomI97RODO8IJAZHELJM?usp=sharing)
[![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/tobifinn/GenSIM)
[![Preprint](https://img.shields.io/badge/Preprint-ArXiv-B31B1B?style=flat&logo=arxiv)](https://arxiv.org/abs/2508.14984)
![Website](https://img.shields.io/badge/Website-Stay_Tuned-lightblue?style=flat)
![Version](https://img.shields.io/badge/Version-0.5-blue?style=flat)

## Repository Structure

``` 
data/
├─ auxiliary – Auxiliary data contained in the repository
├── ds_auxiliary.nc - Auxiliary data file (grid cells, mask)
├── ds_demo.nc - Demo dataset (available at https://doi.org/10.5281/zenodo.17535317)
├─ models - The pre-trained model checkpoints (available at https://huggingface.co/tobifinn/GenSIM)
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
conda env create -f environment.yml
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
- `surrogate.sampler` – Number of steps and schedule parameters for flow matching sampler.
- `surrogate.train_augmentation` – Settings for data augmentation.
- `data` – Paths to data, batch size, number of workers.

You can override any entry from the command line, e.g.:

```bash
python train.py trainer.max_steps=500000 exp_name=my_exp
```

## Training

Ensure the `data/train_data` folder contains the required Zarr files (`train.zarr`, `validation.zarr`) and the auxiliary NetCDF (`auxiliary/ds_auxiliary.nc`).

```bash
python train.py
```

The script logs progress with a tqdm bar, saves checkpoints under `data/models/<exp_name>/`, and (offline) logs to Weights & Biases as configured in `config.yaml`.

To resume training set `ckpt_path` in `config.yaml` to the desired checkpoint.

## Inference

For inference with this repository, we provide pre-trained model checkpoints via [`HuggingFace`](https://huggingface.co/tobifinn/GenSIM).
The model checkpoints define the weights of the neural network and are available in two different versions: either as exponential moving average (`model_weights_ema.safetensors`) or as raw weights (`model_weights.safetensors`).
To avoid a contamination of the weights with malicious data, the model weights are stored in the [`safetensors`](https://huggingface.co/docs/safetensors/en/index) format.

A [`jupyter notebook `](inference_demo.ipynb) is included to showcase prediction steps with GenSIM over a demo dataset.
To use the notebook, ensure the demo dataset `data/auxiliary/ds_demo.nc` is downloaded from [`Zenodo`](https://doi.org/10.5281/zenodo.17535317).
The inference demo initialises GenSIM, loads its checkpoint, and makes ensemble predictions of up to four days.
These predictions are then compared to a persistence forecast and the targetted neXtSIM-OPA simulation.
The code used in the inference notebook can be also used as starting step for other usage.

## Data Layout

Sea‑ice state variables: `sit`, `sic`, `sid`, `siu`, `siv`, `snt`.  
Forcing variables: `tus`, `rhus`, `uas`, `vas`.  

The Zarr file contains a `datacube` array with dimensions `[time, variable, y, x]` and a `var_names` attribute.  

The auxiliary NetCDF provides `mask`, `x_coord`, `y_coord`.

## License

This project is released under the MIT License (see `LICENSE`).

## Citation

If you use GenSIM, please cite the following preprint until publication:

```bibtex
@article{Finn_preprint_2025,
    author={Finn, Tobias Sebastian and Bocquet, Marc and Rampal, Pierre and Durand, Charlotte and Porro, Flavia and Farchi, Alban and Carrassi, Alberto}
    title={Generative AI models enable efficient and physically consistent sea-ice simulations},
    url={http://arxiv.org/abs/2508.14984},
    DOI={10.48550/arXiv.2508.14984},
    note={arXiv:2508.14984 [physics]},
    number={arXiv:2508.14984},
    publisher={arXiv},
    year={2025},
    month=aug
}
```

## Contact

Tobias Sebastian Finn – tobias.finn@enpc.fr

*End of README*
