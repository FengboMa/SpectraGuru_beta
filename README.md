<img src="element/logo.png" width="300">

[![Documentation Status](https://img.shields.io/badge/Documentation-latest-green)](https://fengboma.github.io/docs.spectraguru/) 
[![License: Apache](https://img.shields.io/badge/License-Apache_2.0-yellow)](https://www.apache.org/licenses/LICENSE-2.0) 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://streamlit.io/)

# SpectraGuru - Spectral Analysis Application

*SpectraGuru is currently under development! Thank you for your patience.*

Find our documentation page: [here](https://fengboma.github.io/docs.spectraguru/)!

## What is SpectraGuru?
SpectraGuru is a spectral analysis application designed to provide user-friendly tools for processing and visualizing spectra, aimed at accelerating your research. It functions as a dashboard or a specialized tool within a Python environment, organized with various modular functions that allow users to process spectroscopy data in a pipeline. SpectraGuru is based on the Python Streamlit framework.

![Demo](element/demo.gif)


---

## Quick start

### Visit our site

Our application is hosted [here](https://spectraguru.org)! Please use it directly, as easy as it can go.

### Deploy locally

You do not have to host it locally to use the application. But if you wish to deploy it locally, please follow these steps:

1. Install Python and dependencies

   SpectraGuru is tested with Python 3.12. The important runtime packages are pinned in `requirements.txt`:

   - `streamlit==1.49.1`
   - `pandas==2.3.2`
   - `numpy==2.3.3`
   - `scipy==1.16.2`
   - `scikit-learn==1.7.2`
   - `matplotlib==3.10.6`
   - `seaborn==0.13.2`
   - `altair==5.5.0`
   - `streamlit-extras==0.7.8`
   - `psycopg2==2.9.10`
   - `PyWavelets==1.9.0`
   - `filelock==3.25.2`
   - `deprecation==2.1.0`

   Install them with:

```
pip install -r requirements.txt
```

2. Clone the main repo

```
cd <FILE LOCATION>

git clone https://github.com/FengboMa/SpectraGuru_beta.git
```

3. Run Spectra Application Welcome.py
4. Run the following command

```
streamlit run SpectraGuru_beta/Spectra Application Welcome.py
```
5. Your local version should be up in port 8501 by default in your favorite browser!

---

### About this Project
The project was started in Jun 2024 by Dr. Yiping Zhao and Dr. Xianyan Chen from the University of Georgia. See more [about us](https://www.zhao-nano-lab.com/)!

---

### Help and Support

If you have any questions, comments, and observations, please let us know! Email: zhao-nano-lab@uga.edu.

