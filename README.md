# Drop-Width Estimation from Side-View Videos

<p align="center">
  <img src="The setup.png" alt="experimental setup" width="65%">
</p>

A ready-to-use toolkit to **predict the *front-view* width of a liquid drop** sliding down an inclined surface **using only *side-view* measurements**.  
It reproduces the method published in **Scientific Reports** (2024) and ships the fully-trained Long-Short-Term-Memory (LSTM) network (20-frame sliding window, RMSE 67 µm ≈ 2.4%).

> **Why is this useful?**  
> Drop width enters directly into both the *lateral adhesion* (Furmidge) and *dynamic friction* force equations.  
> Conventional approaches require a second camera or mirror that shrinks the observable area and complicates alignment.  
> Our model keeps the set-up minimal while still giving width data for the **entire 5 cm track**—even when the drop interacts with surface defects.

---

 https://github.com/AK-Berger/Drop_width_estimation/assets/57271994/85117226-5a3e-46f6-be1c-7ed3f92c787b

---

## Quick Start

```bash
# 1️⃣ Clone the repository and create a virtual environment
git clone https://github.com/AK-Berger/Drop_width_estimation.git
cd Drop_width_estimation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Launch the tutorial
jupyter lab tutorial.ipynb
```


---

## Dataset Details (`dataset.xlsx`)

| Column              | Unit    | Description                                |
|---------------------|---------|--------------------------------------------|
| `Status`            | —       | train / test / final_val split             |
| `VideoID`           | —       | ID of the side-view video                  |
| `Frame`             | —       | Frame number within video                  |
| `TiltDeg`           | °       | Surface inclination angle                  |
| `DropLength`        | mm      | Side-view length of the drop               |
| `DropCenterHeight`  | mm      | Drop centroid height                       |
| `AdvAngle`          | °       | Advancing contact angle                    |
| `RecAngle`          | °       | Receding contact angle                     |
| `Velocity`          | mm/s    | Drop sliding speed                         |
| `Width`           | mm      | Ground-truth front-view width              |

- **3,224 frames** from multiple videos recorded at 500 fps.
- Ground-truth width (`Width`) was measured using two mirrors and used only for training and validation.
- `final_val` experiments were acquired externally from the training dataset to assess generalization.

---

## Pre-trained Model (`LSTM_weights.h5`)

- **Architecture**: 3 stacked LSTM layers (64 → 32 → 16) + TimeDistributed dense head.
- **Input**: 20-frame sliding window of 6 numerical features.
- **Output**: Predicted drop width per frame.
- **Loss**: MAE, Optimizer: Adam (lr=1e-4)

Example inference on your own side-view data:

```python
from tensorflow import keras
import numpy as np

features = np.loadtxt('my_new_features.csv', delimiter=',')  # shape: (N_frames, 9)
model = keras.models.load_model('src/model_arch.json', compile=False)
model.load_weights('LSTM_weights.h5')

# Reshape for 20-frame sliding prediction
sliding_input = features.reshape(-1, 20, 9)
predicted_widths = model.predict(sliding_input)
```

---

## Repository Structure

```
Drop_width_estimation/
├── dataset.xlsx            # Side-view measurements and width ground truth
├── LSTM_weights.h5         # Trained model weights
├── tutorial.ipynb          # Step-by-step notebook
├── The setup.png           # Setup image
├── docs/
│   └── demo.mp4            # Optional local copy of the demo video
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Requirements

Install with:

```bash
pip install -r requirements.txt
```

Suggested content of `requirements.txt`:

```
tensorflow==2.12.0
keras==2.12.0
opencv-python>=4.9
numpy>=1.23
pandas>=1.5
scipy>=1.11
matplotlib>=3.8
scikit-learn>=1.3
jupyterlab>=4.0
tqdm>=4.66
```

> For exact replication of paper results, these older versions were used:  
> `tensorflow==2.5.0`, `keras==2.9.0`, `opencv-python==4.5.4`, `numpy==1.20.3`, `pandas==1.3.4`, `scipy==1.7.1`, `matplotlib==3.4.3`.

---

## Citation

If you use this repository, please cite the following article:

```bibtex
@article{Shumaly2024,
  title   = {Estimating sliding drop width via side-view features using recurrent neural networks},
  author  = {Shumaly, Sajjad and Darvish, Fahimeh and Li, Xiaomei and Kukharenko, Oleksandra and Steffen, Werner and Guo, Yanhui and Butt, Hans-Jürgen and Berger, Rüdiger},
  journal = {Scientific Reports},
  volume  = {14},
  pages   = {12033},
  year    = {2024},
  doi     = {10.1038/s41598-024-62194-w}
}
```

---

## Contact

For questions or collaborations:

- **Technical**: Sajjad Shumaly — <shumalys@mpip-mainz.mpg.de>  
- **Conceptual**: Rüdiger Berger — <berger@mpip-mainz.mpg.de>

---

<div align="center">
  <em>Happy (single-camera) sliding-drop experiments! 🚀</em>
</div>
