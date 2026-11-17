# Walk / Run Activity Recognition

An interactive Streamlit web app that uses an XGBoost classifier to predict whether a sensor record represents walking or running.

## Project files

- `app.py` — Streamlit web application (single-row and CSV batch predictions).
- `train_model.py` — Re-trains the XGBoost model from `walkrun.csv` and saves `model.pkl`.
- `model.pkl` — Pre-trained model bundle, ready for deployment.
- `walkrun.csv` — Dataset used by the original notebook.
- `Walk_Run.ipynb` — Original notebook with one additional final cell to export the already-trained `xgb` model.
- `requirements.txt` — Python dependencies.

## Run locally

1. Install Python.
2. Open a terminal in this project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Retrain the model (optional)

Keep `walkrun.csv` in the same folder, then run:

```bash
python train_model.py
```

This writes a new `model.pkl`. The supplied `model.pkl` is already trained, so retraining is not required to launch the app.

## Deploy with GitHub and Streamlit Community Cloud

1. Create a **Public** GitHub repository.
2. Upload all files from this folder, preserving the file names and folder structure.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
5. Deploy. Streamlit installs packages from `requirements.txt`.

## Input columns

The model expects these columns in this exact order:

`wrist`, `acceleration_x`, `acceleration_y`, `acceleration_z`, `gyro_x`, `gyro_y`, `gyro_z`

For batch predictions, upload a CSV containing all seven columns. Additional columns are ignored.

## Dataset and evaluation notes

The original notebook drops `date`, `time`, and `username`, applies IQR-based median replacement to the remaining columns, then performs an 80:20 random split with `random_state=45`. This project keeps that workflow for consistency with the supplied notebook.

The supplied hold-out score is a result from that particular random split, not a guarantee of performance on new people or future activity sessions. Since sensor data can be sequential, a chronological or user/session-based validation should be considered before making real-world performance claims.

The form accepts individual sensor rows. It does not connect to a smartwatch or stream live sensor data.

## Target labels

- `0` = Walking
- `1` = Running

The dataset encodes `wrist` as 0/1, but the provided source does not document which encoded value corresponds to left or right wrist.
