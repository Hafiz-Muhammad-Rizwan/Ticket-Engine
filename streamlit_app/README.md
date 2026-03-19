# Hybrid Ticket Intelligence - Streamlit App

## 1. What This App Does
- Takes a new ticket description as input.
- Uses hybrid retrieval with alpha blending between keyword (TF-IDF) and semantic similarity.
- Shows predicted ticket type and top similar resolutions.
- Provides side-by-side model behavior insights.

## 2. Folder Structure
- `streamlit_app/app.py` - main app
- `streamlit_app/requirements.txt` - dependencies
- `streamlit_app/artifacts/` - optional exported notebook artifacts

## 3. Recommended Artifacts From Kaggle
Copy these files from Kaggle `/kaggle/working/artifacts` into `streamlit_app/artifact/`.
The app also supports `streamlit_app/artifacts/` for compatibility:
- `metadata.json`
- `idf.npy`
- `embed_matrix.npy`
- `dense_doc_vectors.npy`
- `corpus.csv`
- `stoi.json`

If artifacts are missing, the app still runs using fallback semantic vectors built from local CSV.

## 4. Run Locally
From `streamlit_app` folder:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## 5. Notes
- The app auto-detects columns from the dataset if needed.
- Missing or NaN resolutions are replaced with `No resolution available`.
