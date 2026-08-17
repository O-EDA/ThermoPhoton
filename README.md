# ThermoPhoton

PyTorch implementation of the Transformer operator model from:

> W. Guan, L. Huang, Y. Lin, Y. Wu, Y. Tong, and Y. Ma,
> "ThermoPhoton: Fast 3D Thermal Simulation of Photonic Integrated Circuits
> via Operator Learning," ICCAD 2025.

ThermoPhoton learns steady-state 3D temperature fields from 2D heater
distributions using physics losses only. This release contains the training
code, the paper checkpoint, COMSOL models and full-field data, and a complete
inference example.

## Citation

```bibtex
@INPROCEEDINGS{Guan2025ThermoPhoton,
  author={Guan, Weilong and Huang, Li and Lin, Yuxuan and Wu, Yuchao and Tong, Yeyu and Ma, Yuzhe},
  booktitle={2025 IEEE/ACM International Conference on Computer Aided Design (ICCAD)},
  title={ThermoPhoton: Fast 3D Thermal Simulation of Photonic Integrated Circuits via Operator Learning},
  year={2025},
  pages={1-8},
  doi={10.1109/ICCAD66269.2025.11240705},
}
```

## Installation

Python 3.9--3.12 and a CUDA-capable PyTorch installation are recommended.
The COMSOL archive is tracked with Git LFS, so Git LFS must be installed before
cloning or pulling the repository.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage

Run full-field inference with the included checkpoint:

```bash
python main.py \
  --checkpoint checkpoints/thermophoton.pt \
  --output-dir outputs/evaluation
```

The `create_example_heat_source` call in `main.py` can be replaced with any
flattened 2D heater grid of the same resolution.

Train with the paper schedule (20k Adam steps followed by 80k AdamW steps):

```bash
python main.py --train-new-model --checkpoint outputs/thermophoton.pt
```

## Repository layout

```text
ThermoPhoton/
├── checkpoints/thermophoton.pt # released Transformer DeepONet weights
├── data/comsol_ground_truth.7z # COMSOL models and full 3D fields
├── main.py                    # training and 3 x 3 MZI field inference
├── src/thermophoton/network.py # sampler, Transformer branch, Fourier trunk
├── src/deepheat/              # minimal PyTorch/ZCS runtime
└── tests/                     # release smoke tests
```

`data/comsol_ground_truth.7z` contains the COMSOL projects, exported temperature
point clouds, and saved ThermoPhoton prediction fields for 3 x 3 MRR, 3 x 3
MZI, 4 x 4 MZI, and random blocks. The 120 x 120 x 120 fields contain 1,728,000
points each; the archive also includes the 200-grid random-block export.

Each text row stores `X Y Z Temperature`. COMSOL exports use coordinates on
approximately `[0, 1000]` and temperature in kelvin. ThermoPhoton prediction
files use normalized `[0, 1]` coordinates and temperature in degrees Celsius.

The included `.mph` projects were saved by COMSOL 6.2.0.290 and require a
compatible COMSOL installation with the CAD Import Module to open or rerun.
COMSOL is not required to read the exported text fields or run the model.

## License

ThermoPhoton-specific code, models, and data are released under the
[BSD 3-Clause License](LICENSE).
The bundled `src/deepheat` runtime is derived from DeepXDE 1.13.1 and remains
subject to LGPL-2.1.
