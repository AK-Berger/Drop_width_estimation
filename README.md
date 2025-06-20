# Drop-Width Estimation from Side-View Videos

<p align="center">
  <img src="The setup.png" alt="experimental setup" width="65%">
</p>



A Python toolkit to estimate the **front-view width** of a liquid drop sliding down an inclined surface using only **side-view** measurements.\
Implementation of Shumaly *et al.*, **Scientific Reports** (2024).

> **Why is this useful?**\
> Drop width enters directly into both the *lateral adhesion* (Furmidge) and *dynamic friction* force equations.\
> Conventional approaches require a second camera or mirror that shrinks the observable area and complicates alignment.\
> Our model keeps the set-up minimal while still giving width data for the **entire 5 cm track**—even when the drop interacts with surface defects.

---

[https://github.com/AK-Berger/Drop\_width\_estimation/assets/57271994/85117226-5a3e-46f6-be1c-7ed3f92c787b](https://github.com/AK-Berger/Drop_width_estimation/assets/57271994/85117226-5a3e-46f6-be1c-7ed3f92c787b)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Data Structure](#data-structure)
- [Model Architecture](#model-architecture)
- [Output](#output)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Features

- Train an LSTM-based regression model on side-view drop features.
- Predict drop width from new video data for any **Video ID**.
- Automatic normalization, sliding-window slicing, and result plotting.
- Example scripts: `train_model.py` and `inference.py`.

## Installation

```bash
git clone https://github.com/AK-Berger/Drop_width_estimation.git
cd Drop_width_estimation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train_model.py \
  --data-file data/dataset.xlsx \
  --model-output models/lstm_weights.h5 \
  --epochs 2500 \
  --batch-size 16 \
  --window-size 20
```

- **--data-file**: Path to the Excel dataset (must include `status`).
- **--model-output**: Path to save trained model weights (.h5).
- **--epochs**, **--batch-size**, **--window-size**: Training parameters.

### Inference

```bash
python inference.py \
  --data-file data/dataset.xlsx \
  --weights-file models/lstm_weights.h5 \
  --video-id 14 \
  --window-size 20 \
  --output output/14.png
```

- **--data-file**: Source Excel (with `Video ID`, features, `status`).
- **--weights-file**: Trained LSTM weights (.h5).
- **--video-id**: Integer ID of the video to process.
- **--window-size**: Sliding window length (frames).
- **--output**: (Optional) Path to save plot (PNG). Without this, the plot displays interactively.
- **Results Excel**: Saved automatically to `output/<datafile_stem>_<video_id>.xlsx`.

## Data Structure

### Input: `data/dataset.xlsx`

| Column                                       | Unit | Description                            |
| -------------------------------------------- | ---- | -------------------------------------- |
| `status`                                     | —    | `train`, `test`, or `final validation` |
| `Video ID`                                   | —    | Unique video identifier                |
| `Frame`                                      | —    | Frame number                           |
| `Advancing (degree)`                         | °    | Advancing contact angle                |
| `Receding (degree)`                          | °    | Receding contact angle                 |
| `Drop length (cm)`                           | cm   | Side-view drop length                  |
| `Drop height (cm)`                           | cm   | Drop centroid height                   |
| `Velocity (cm/s)`                            | cm/s | Drop sliding speed                     |
| `Middle line angle (degree)`                 | °    | Angle of drop midline                  |
| `Tilt angle (degree)`                        | °    | Inclination of the surface             |
| `Defect size [thickness,length,height] (μm)` | μm   | Surface defect dimensions              |
| `Drop width (cm)`                            | cm   | Ground-truth front-view width          |

### Intermediate Outputs: `output/`

- `dataset_<video_id>.xlsx`: Filtered data for each Video ID (e.g., `dataset_14.xlsx`, `dataset_234.xlsx`).

## Model Architecture

- **LSTM layer**: 48 units, tanh activation, L2 regularization (λ=0.01)
- **Dropout**: 0.5
- **Dense output**: 1 unit, tanh activation
- **Input shape**: `(window_size, 6)` features per frame
- **Loss**: MSE, **Optimizer**: Adam
- **Label scaling**: Ground-truth width scaled from cm to μm

## Output

- **Excel**: `output/<datafile_stem>_<video_id>.xlsx` containing original features plus:
  - `Estimated Width` (μm)
  - `Drop width (cm)` (measured)
- **Plot**: Overlaid estimated vs measured widths vs frame index

## Project Structure

```
Drop_width_estimation/
├── data/
│   └── dataset.xlsx
├── models/
│   └── lstm_weights.h5
├── output/
│   ├── dataset_14.xlsx
│   └── dataset_234.xlsx
├── src/
│   └── drop_width/
│       ├── preprocessing.py
│       ├── model.py
│       └── side_to_width.py
├── train_model.py
├── inference.py
├── requirements.txt
├── The setup.png
└── README.md
```

## Requirements

See `requirements.txt` for full package versions.

## License

Licensed under the GNU General Public License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{Shumaly2024,
  title   = {Estimating sliding drop width via side-view features using recurrent neural networks},
  author  = {Shumaly, Sajjad and Darvish, Fahimeh and Li, Xiaomei and Kukharenko, Oleksandra and Steffen, Werner and Guo, Yanhui and Butt, Hans-J{"u}rgen and Berger, R{"u}diger},
  journal = {Scientific Reports},
  volume  = {14},
  pages   = {12033},
  year    = {2024},
  doi     = {10.1038/s41598-024-62194-w}
}
```

## Contact

- **Technical**: Sajjad Shumaly — [shumalys@mpip-mainz.mpg.de](mailto\:shumalys@mpip-mainz.mpg.de)
