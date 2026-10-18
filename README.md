# STRiDE: Real-Time Gait Tracking for Parkinson's Disease Assessment

STRiDE is an end-to-end framework for assessing Parkinson's disease (PD) gait impairment from ground reaction force (GRF) data. The project combines gait feature analysis, image-based deep learning, a low-cost pressure-sensing insole, and a web application for continuous motor behavior monitoring.

---

## How It Works

GRF signals are first cleaned and converted into gait measures related to timing, walking strength, postural stability, and movement coordination. The same signals are also transformed into Gramian Angular Field images so temporal gait patterns can be represented spatially.

STRiDE uses a Spatiotemporal Generative Adversarial Network (ST-GAN) to generate additional gait representations and classify four severity groups: Healthy, Slight PD, Mild PD, and Moderate PD. A wearable insole with force-sensitive resistors and an Arduino-compatible microcontroller collects pressure data for real-time analysis, while the web application displays gait signals, pressure patterns, gait metrics, and model predictions.

---

## Components

**1. Data Preprocessing** — `data_preprocessing.py`

Removes the first and last 10 seconds of each GRF recording, sets force values below 10 N to zero, and prepares the recordings for downstream analysis.

**2. Gait Feature Extraction** — `gait_features.py`

Extracts gait measures related to cadence, stride timing, stance time, peak force, force impulse, center of pressure, and left-right asymmetry.

**3. Gait Feature Analysis** — `feature_analysis.py`

Runs one-way ANOVA across PD severity groups to identify gait features associated with differences in motor behavior.

**4. GAF Transformation** — `gaf_transform.py`

Transforms left and right GRF signals into Gramian Angular Difference Field (GADF) or Gramian Angular Summation Field (GASF) images for image-based classification.

**5. ST-GAN** — `model.py`

Defines the generator and discriminator/classifier used to augment gait representations and classify PD severity.

**6. Training** — `train.py`

Trains the ST-GAN using GAF images and saves the best generator and discriminator checkpoints.

**7. Evaluation** — `evaluate.py`

Calculates accuracy, precision, recall, F1 score, the confusion matrix, and class-level performance.

**8. Baseline Comparison** — `baselines.py`

Compares ST-GAN with CNN models trained on GADF and GASF images and with Random Forest and SVM models trained on extracted gait features.

**9. Sensor Processing** — `sensor_processing.py`

Converts raw force-sensitive resistor readings into resistance, conductance, and estimated weight using the sensor calibration equations used in the wearable prototype.

**10. Web Application** — `app.py`

Displays GRF signals, pressure patterns, and predicted PD severity from uploaded gait recordings.

---

## Results

The strongest configuration used **GADF images with ST-GAN** and achieved 98% accuracy on the test dataset.

| Representation | Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---:|---:|---:|---:|
| **GADF** | **ST-GAN** | **98.0%** | **98.1%** | **98.0%** | **98.0%** |
| GADF | CNN | 88.6% | 81.6% | 88.6% | 84.9% |
| GASF | ST-GAN | 93.8% | 94.3% | 93.8% | 92.4% |
| GASF | CNN | 90.9% | 83.9% | 90.9% | 87.2% |
| Grayscale | ST-GAN | 87.9% | 89.4% | 87.9% | 86.0% |
| Grayscale | CNN | 84.3% | 78.6% | 84.3% | 81.0% |
| Gait Features | Random Forest | 67.7% | 71.4% | 67.7% | 67.4% |
| Gait Features | SVM | 64.5% | 65.4% | 64.5% | 64.4% |

Class-level performance for the GADF ST-GAN model:

| Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Healthy | 97.9% | 100.0% | 98.9% |
| Slight PD | 100.0% | 98.2% | 99.1% |
| Mild PD | 95.3% | 100.0% | 97.6% |
| Moderate PD | 100.0% | 83.3% | 90.9% |

---

## Data

STRiDE uses ground reaction force recordings from three gait datasets containing healthy participants and people with Parkinson's disease. Each recording contains pressure measurements from sensors under the left and right feet together with total GRF signals.

The preprocessing code expects a metadata file with the following columns:

```text
subject_id,file,severity
Ga01,Ga01_01.txt,0
Ga02,Ga02_01.txt,1
```

Severity labels are encoded as:

```text
0 = Healthy
1 = Slight PD
2 = Mild PD
3 = Moderate PD
```

The raw GRF text files are expected to contain time, eight left-foot sensor channels, eight right-foot sensor channels, and left/right total force columns.

---

## Wearable System

The prototype uses force-sensitive resistors embedded in an insole, an Arduino-compatible microcontroller, an FPC converter board, resistors, and a microSD module. The sensor readings are converted to force measurements and can be passed into the same gait analysis pipeline used for the GRF datasets.

The prototype system cost is under $70.

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: PyTorch, NumPy, pandas, SciPy, scikit-learn, Matplotlib, and Streamlit.

---

## Usage

### 1. Preprocess GRF recordings

```bash
python data_preprocessing.py \
  --raw-dir data/raw \
  --metadata data/metadata.csv \
  --out-dir data/processed
```

### 2. Extract gait features

```bash
python gait_features.py --processed-dir data/processed
```

### 3. Run gait feature analysis

```bash
python feature_analysis.py
```

### 4. Build GADF images

```bash
python gaf_transform.py \
  --processed-dir data/processed \
  --out-dir data/gadf \
  --method gadf
```

To generate GASF images for comparison:

```bash
python gaf_transform.py \
  --processed-dir data/processed \
  --out-dir data/gasf \
  --method gasf
```

### 5. Train ST-GAN

```bash
python train.py \
  --image-dir data/gadf \
  --out-dir results/stgan
```

### 6. Evaluate the model

```bash
python evaluate.py \
  --image-dir data/gadf \
  --checkpoint results/stgan/discriminator_best.pt
```

### 7. Run baseline models

```bash
python baselines.py
```

### 8. Launch the web application

```bash
streamlit run app.py
```

---

## Repository Structure

```text
STRiDE/
├── data_preprocessing.py
├── gait_features.py
├── feature_analysis.py
├── gaf_transform.py
├── model.py
├── train.py
├── evaluate.py
├── baselines.py
├── sensor_processing.py
├── app.py
├── requirements.txt
└── arduino/
    └── stride_sensor.ino
```


