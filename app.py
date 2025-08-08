# app.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import numpy as np
import io

# ======================
# Initialisation de l'API
# ======================
app = FastAPI(title="API de prédiction de faux billets")


origins = [
    "https://appapp-nfpcetlczdmgrygg4skk2j.streamlit.app/",  # ton app Streamlit
    "https://fastapi-api-54e7.onrender.com",  # optionnel
    "*"  # à enlever en prod pour plus de sécurité
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ======================
# Chargement des modèles
# ======================
model = joblib.load("logistic_regression_model.sav")
scaler = joblib.load("standard_scaler.sav")

# ======================
# Fonction : suppression des outliers
# ======================
def supprimer_outliers(df, colonnes):
    for colonne in colonnes:
        Q1 = df[colonne].quantile(0.25)
        Q3 = df[colonne].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[colonne] >= lower_bound) & (df[colonne] <= upper_bound)]
    return df

# ======================
# Endpoint principal
# ======================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire le contenu du fichier CSV
        contents = await file.read()
        new_data = pd.read_csv(io.StringIO(contents.decode("utf-8")), sep=';')

        # Vérification des colonnes attendues
        expected_cols = ['length', 'height_left', 'height_right', 'margin_up', 'margin_low', 'diagonal', 'is_genuine']
        if not set(expected_cols).issuperset(new_data.columns):
            return JSONResponse(
                status_code=400,
                content={"error": f"Colonnes attendues : {expected_cols}"}
            )


        # Suppression de la colonne 'is_genuine'
        if 'is_genuine' in new_data.columns:
            new_data = new_data.drop(columns=['is_genuine'])

        # Traitement des valeurs manquantes
        if new_data["margin_low"].isnull().sum() > 0:
            mediane = new_data["margin_low"].median()
            new_data["margin_low"] = new_data["margin_low"].fillna(mediane)

        # Suppression des outliers
        colonnes = ['length', 'height_left', 'height_right', 'margin_up', 'margin_low', 'diagonal']
        new_data = supprimer_outliers(new_data, colonnes)

        # Standardisation
        new_data_scaled = scaler.transform(new_data)

        # Prédiction
        predictions = model.predict(new_data_scaled)

        # Construction des résultats
        result = new_data.copy()
        result["prediction"] = predictions
        #result["prediction_label"] = result["prediction"].map({0: "Faux billet", 1: "Vrai billet"})

        result = result.replace({np.nan: None, np.inf: None, -np.inf: None})
        
        # Retour JSON
        return {
             "predictions": result.to_dict(orient="records")
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
