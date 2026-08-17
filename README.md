# ThermoPhoton

PyTorch implementation of the Transformer operator model from:

> W. Guan, L. Huang, Y. Lin, Y. Wu, Y. Tong, and Y. Ma,
> "ThermoPhoton: Fast 3D Thermal Simulation of Photonic Integrated Circuits
> via Operator Learning," ICCAD 2025.

ThermoPhoton learns steady-state 3D temperature fields from 2D heater
distributions using physics losses only. This release contains the training
code, the paper checkpoint, COMSOL models and full-field data for the four
released validation cases, compact table references, and inference and
validation entry points.

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

Reproduce the four released COMSOL comparison rows directly from the included
checkpoint:

```bash
python validate.py
```

The validation samples the paper-defined heater regions on the 120 x 120 x 120
grid around `z = 0.5 mm`, recomputes the mean absolute temperature error and
MAPE, and checks that inference agrees with the archived predictions within
0.005 K. Expected CPU results are:

| Case | Mean absolute error (K) | MAPE (%) |
|---|---:|---:|
| 3 x 3 MRR | 0.444 | 0.145 |
| 3 x 3 MZI | 0.109 | 0.036 |
| 4 x 4 MZI | 0.151 | 0.050 |
| Random blocks | 0.216 | 0.072 |

Train with the paper schedule (20k Adam steps followed by 80k AdamW steps):

```bash
python main.py --train-new-model --checkpoint outputs/thermophoton.pt
```

Evaluate an existing checkpoint and save temperature slices:

```bash
python main.py \
  --checkpoint checkpoints/thermophoton.pt \
  --output-dir outputs/evaluation
```

## Repository layout

```text
ThermoPhoton/
├── checkpoints/thermophoton.pt # released Transformer DeepONet weights
├── data/comsol_ground_truth.7z # COMSOL models and full 3D fields
├── data/table1/                # compact per-heater references
├── main.py                    # training and 3 x 3 MZI field inference
├── validate.py                # checkpoint-to-COMSOL table validation
├── src/thermophoton/network.py # sampler, Transformer branch, Fourier trunk
├── src/deepheat/              # minimal PyTorch/ZCS runtime
└── tests/                     # release smoke tests
```

`data/comsol_ground_truth.7z` contains the COMSOL projects, exported temperature
point clouds, and saved ThermoPhoton prediction fields for 3 x 3 MRR, 3 x 3
MZI, 4 x 4 MZI, and random blocks. The 120 x 120 x 120 fields contain 1,728,000
points each; the archive also includes the 200-grid random-block export.

Each text row stores `X Y Z Temperature`. COMSOL exports use coordinates on
approximately `[0, 1000]` and temperature in kelvin; validation divides the
coordinates by 1000. ThermoPhoton prediction files use normalized `[0, 1]`
coordinates and temperature in degrees Celsius; validation adds 273.15 to
compare in kelvin.

The compact JSON files preserve the heater-region statistics used by
`validate.py`, allowing checkpoint validation without unpacking the multi-GB
archive. The included `.mph` projects were saved by COMSOL 6.2.0.290 and require
a compatible COMSOL installation with the CAD Import Module to open or rerun.
COMSOL is not required to read the exported text fields or run the included
model.

## License

ThermoPhoton-specific code, models, and data are released under the
[BSD 3-Clause License](LICENSE).
The bundled `src/deepheat` runtime is derived from DeepXDE 1.13.1 and remains
subject to LGPL-2.1.
