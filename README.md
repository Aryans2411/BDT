# SENTINEL - Real-Time Biometric Authentication Dashboard

**Version 5.3 (HWiNFO Integration)**

An extensible dashboard for real-time ECG biometric authentication and hardware monitoring, integrating Spark-based analytics, ML pipeline visualization, and live system metrics (including CPU temperature via HWiNFO).

---

## 🚀 Features

- **Real-Time Authentication:** Classifies ECG signals for secure biometric access—distinguishing between real users and multiple intruder profiles.
- **Big Data Ready:** Utilizes Apache Spark for scalable, parallelized data processing.
- **Live System Monitoring:** Displays CPU temperature, RAM usage, and per-core CPU utilization, leveraging HWiNFO (with `py-hwinfo`).
- **Deep ML Analysis:** Shows PCA spaces, feature importance, detailed classification reports, and allows live-signal walkthroughs.
- **Visualization Suite:** Interactive, Streamlit-powered UI with advanced plotting for PCA, FFT, raw waveforms, and more.
- **Modular Data Loading:** Supports easy swapping and augmentation of user/intruder datasets.

---

## 🖥️ Screenshots

_(Add screenshots here, e.g., from your Streamlit dashboard, PCA plots, live hardware stats, etc.)_

---

## 📂 Project Structure

sentinel/
│
├── main_app.py # Main Streamlit application (this file)
├── augmented_real_user_20k.csv
├── augmented_intruder_1_20k.csv
├── augmented_intruder_2_20k.csv
├── augmented_intruder_3_20k.csv
├── heatplot_better.png
├── README.md
└── requirements.txt

---

## ⚙️ Requirements

- **Python 3.8+**
- **pip** (Python package manager)
- **HWiNFO** (Windows system monitoring tool)
  - HWiNFO must be installed and running in the background
  - Enable “Shared Memory Support” in HWiNFO Settings

#### Python Libraries

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `findspark`
- `pyspark`
- `psutil`
- `py-hwinfo`
- `scikit-learn`

Install requirements via:

```bash
pip install -r requirements.txt

```

## Run the app

```bash
streamlit run try5.py
```
