# ThermoPhoton

Implementation of the ThermoPhoton 3D thermal simulation framework proposed in
ICCAD 2025.

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
  --config data/3x3_mzi.json \
  --output-dir outputs/evaluation
```

Case geometries are defined in `data/`.

## Training

Run training with the default configuration:

```bash
python train.py
```

## Repository layout

```text
ThermoPhoton/
├── checkpoints/thermophoton.pt # released ThermoPhoton weights
├── data/*.json                # heater configurations
├── data/comsol_ground_truth.7z # COMSOL models and full 3D fields
├── deepheat/                  # training framework
├── branch_networks.py         # training network
├── main.py                    # full-field inference
├── train.py                   # training
├── src/thermophoton/network.py # ThermoPhoton network
└── tests/                     # release smoke tests
```

`data/comsol_ground_truth.7z` contains the COMSOL projects, exported temperature
point clouds, and saved ThermoPhoton prediction fields.

Each text row stores `X Y Z Temperature`. COMSOL exports use coordinates on
approximately `[0, 1000]` and temperature in kelvin. ThermoPhoton prediction
files use normalized `[0, 1]` coordinates and temperature in degrees Celsius.

The included `.mph` projects were saved by COMSOL 6.2.0.290 and require a
compatible COMSOL installation with the CAD Import Module to open or rerun.
COMSOL is not required to read the exported text fields or run the model.

## License

ThermoPhoton-specific code, models, and data are released under the
[BSD 3-Clause License](LICENSE).
