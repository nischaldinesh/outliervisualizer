# OutlierVisualizer

A **comparative dashboard** for visualizing and comparing the outputs of multiple outlier detection algorithms on your dataset, built with Streamlit.

---

## Features

- **Interactive Dataset Selection**: Choose from pre‑loaded datasets or add your own CSV files.
- **Algorithm Selection**: Pick up to two algorithms per category (Proximity‑based, Probabilistic, Ensembles, Linear Models).
- **Heatmap Visualization**: Display raw, thresholded, interpolated, binary, or ranked outlier scores across a 2D PCA projection.
- **Custom Colormaps**: Preview and apply multiple Matplotlib colormaps.
- **One‑click Deployment**: Instructions for running locally or hosting on Streamlit Community Cloud.

---

## Prerequisites

- **Python 3.8+**
- **Git**
- **pip** (or **pipenv** / **poetry**)

---

## Clone the Repository

```bash
# replace <username> with your GitHub handle
git clone https://github.com/<username>/outliervisualizer.git
cd outliervisualizer
```

---

## Setup & Installation

1. **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```

2. **Install dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Add your datasets**
    - Place any CSV files in the `datasets/` folder. Filenames should match the keys used in the app (e.g. `Bank.csv`, `Iris.csv`, etc.).

---

## Running Locally

Start the Streamlit app:

```bash
streamlit run app.py
```

Open your browser to [http://localhost:8501](http://localhost:8501) to explore the dashboard.

---

## Project Structure

```
outliervisualizer/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── datasets/             # Example CSV datasets
│   ├── Bank.csv
│   └── Iris.csv
├── utils/                # Helper modules
│   ├── algorithm_runner.py
│   └── dataset_loader.py
└── README.md             # This file
```

---

## Deploying to Streamlit Community Cloud

1. Push your code to GitHub (ensure `requirements.txt` and `datasets/` are included).
2. Sign in at [https://share.streamlit.io](https://share.streamlit.io).
3. Click **New app**, select your repo and branch, and point at `app.py`.
4. Click **Deploy** and share your live URL.

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

