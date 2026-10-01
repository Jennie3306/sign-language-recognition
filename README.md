# Sign Language Recognition via Webcam

**Real-time Sign Language Alphabet Recognition via Webcam using Hand Landmarks**

Capstone project (Đồ án tổng hợp) – Artificial Intelligence track, semester 261.

The system recognizes the **24 static letters** of the ASL (American Sign Language) fingerspelling alphabet in real time from a webcam. J and Z are out of scope because they require motion. It uses MediaPipe to extract 21 hand landmarks, normalizes them into a 63-dimensional feature vector, and compares three self-trained classifiers: **SVM**, **Random Forest**, and an **MLP** (TensorFlow/Keras).

> 🚧 Work in progress. See [Progress](#progress).

---

## Pipeline

```text
Webcam frame / Kaggle image
        │
        ▼
MediaPipe Hand Landmarker ──► 21 landmarks (x, y, z)
        │
        ▼
Normalization ──► 63-dim vector (wrist origin, scaling, left-hand mirroring)
        │
        ├──► SVM (RBF kernel)
        ├──► Random Forest
        └──► MLP (TensorFlow/Keras)
                │
                ▼
     Evaluation: accuracy, macro F1, confusion matrix, FPS
                │
                ▼
     Real-time demo: prediction smoothing + word building
```

## Project Structure

```text
sign-language-recognition/
├── data/
│   ├── raw/              # Kaggle dataset images (not tracked by Git)
│   ├── landmarks/        # extracted landmark CSV files
│   └── collected/        # self-collected webcam data
├── models/
│   └── hand_landmarker.task   # MediaPipe model (download, see Installation)
├── notebooks/            # extraction, training, and evaluation notebooks
├── reports/figures/      # figures for the report
├── src/
│   ├── hand_tracker.py   # HandTracker class wrapping the MediaPipe Tasks API, skeleton drawing
│   └── tests/
│       └── test_webcam.py    # webcam and hand detection check
├── requirements.txt
└── README.md
```

## Installation

**Requirements:** Python 3.11 and a webcam. No GPU needed.

1. Clone the repo and create a virtual environment:

   ```powershell
   git clone https://github.com/Jennie3306/sign-language-recognition.git
   cd sign-language-recognition
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
   ```

2. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. The MediaPipe Hand Landmarker model is already included in `models/`.
   If it is missing, download it with:

   ```powershell
   Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "models\hand_landmarker.task"
   ```

   macOS/Linux:

   ```bash
   curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   ```

## Usage

Run all commands from the **repository root**.

**Check the webcam and hand detection:**

```powershell
python src/tests/test_webcam.py
```

A webcam window shows the 21-point hand skeleton and whether the hand is left or right. Press `q` to quit.

## Data

| Source | Description | Role |
| --- | --- | --- |
| [ASL Alphabet (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | 200×200 images, 29 classes; 24 static letters used | Training |
| Self-collected via webcam | Landmarks recorded directly, tagged with signer and session | Unseen-signer testing |

Download the Kaggle dataset and extract it into `data/raw/`. This folder is not tracked by Git because of its size.

## Evaluation

All models are compared on the same test set using accuracy, macro F1-score, confusion matrix, inference time, and accuracy on unseen signers.

| Model | Accuracy (test) | Macro F1 | Accuracy (unseen signers) | ms/sample |
| --- | --- | --- | --- | --- |
| SVM | – | – | – | – |
| Random Forest | – | – | – | – |
| MLP | – | – | – | – |

*Results will be updated once training is complete.*

## Progress

- [x] Environment setup, migrated to the MediaPipe Tasks API
- [x] Hand detection and skeleton drawing via webcam
- [ ] Extract and normalize landmarks from the Kaggle dataset
- [ ] Collect webcam data
- [ ] Train SVM, Random Forest, and MLP
- [ ] Evaluate and analyze results
- [ ] Real-time demo: smoothing, word building, model switching

## Tech Stack

Python · MediaPipe · OpenCV · NumPy · pandas · scikit-learn · TensorFlow/Keras · Matplotlib · seaborn · Jupyter

## References

1. N. Pugeault, R. Bowden. *Spelling it out: Real-time ASL fingerspelling recognition.* ICCV Workshops, 2011.
2. B. Kang, S. Tripathi, T. Q. Nguyen. *Real-time sign language fingerspelling recognition using convolutional neural networks from depth map.* [arXiv:1509.03001](https://arxiv.org/abs/1509.03001), 2015.
3. F. Zhang et al. *MediaPipe Hands: On-device Real-time Hand Tracking.* [arXiv:2006.10214](https://arxiv.org/abs/2006.10214), 2020.
4. C. Cortes, V. Vapnik. *Support-vector networks.* Machine Learning, 20(3), 1995.
5. L. Breiman. *Random forests.* Machine Learning, 45(1), 2001.

## Author

**Lê Như Nhã Uyên** – Capstone project, semester 261.