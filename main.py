from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import date, datetime
import pandas as pd
import numpy as np
import os
import json
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials

# ID du dossier Drive (remplace par ton vrai ID)
DRIVE_FOLDER_ID = "1XXcOiXZX80AwsyGFkR1UCY3h9hfThGT4"

def upload_to_drive(filepath, filename):

    credentials = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )

    service = build("drive", "v3", credentials=credentials)

    file_metadata = {
        "name": filename,
        "parents": ["1XXcOiXZX80AwsyGFkR1UCY3h9hfThGT4"]
    }

    media = MediaFileUpload(filepath, resumable=True)

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    return file.get("id")


# =========================================================
# APP & DOSSIERS
# =========================================================

app = FastAPI()

UPLOAD_DIR = Path("uploads")
EXPORT_DIR = Path("exports")
UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

# =========================================================
# CONFIG CAPACITÉS
# =========================================================

class ChargeConfig(BaseModel):
    charge_max_1m: float = 60
    charge_max_3m: float = 130
    charge_max_6m: float = 210

CHARGE_CONFIG = ChargeConfig()

# =========================================================
# UTILS
# =========================================================

def sanitize_json(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return obj

def normalize_str(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()

def parse_date_safe(x):
    try:
        d = pd.to_datetime(x, errors="coerce")
        return d.date() if not pd.isna(d) else None
    except Exception:
        return None

def days_until(d):
    if d is None:
        return None
    return (d - date.today()).days

# =========================================================
# RÈGLES MÉTIER (VALIDÉES)
# =========================================================

SEGMENTATION_MAP = {
    "stratégique": 4,
    "strategique": 4,
    "projet": 3,
    "réassort": 2,
    "reassort": 2,
    "à développer": 1,
    "a développer": 1,
    "a developper": 1,
}

TRANSFO_MAP = {
    20: 0.70, 40: 0.85, 60: 1.00, 80: 1.15,
    0.2: 0.70, 0.4: 0.85, 0.6: 1.00, 0.8: 1.15,
    "20%": 0.70, "40%": 0.85, "60%": 1.00, "80%": 1.15,
}

AVG_COMPLEXITE = 2.5
AVG_SEGMENTATION = 2.5
AVG_TRANSFO = 0.925

def get_complexite(v):
    try:
        v = float(v)
        return v if 1 <= v <= 4 else AVG_COMPLEXITE
    except Exception:
        return AVG_COMPLEXITE

def get_segmentation(v):
    if pd.isna(v):
        return AVG_SEGMENTATION
    return SEGMENTATION_MAP.get(normalize_str(v), AVG_SEGMENTATION)

def get_transfo(v):
    if pd.isna(v):
        return AVG_TRANSFO
    return TRANSFO_MAP.get(v, AVG_TRANSFO)

def coeff_urgence(days_left, horizon):
    if days_left is None:
        return 1.0
    if days_left < 0:
        return 1.5

    if horizon == 30:
        return 1.4 if days_left <= 7 else 1.25 if days_left <= 14 else 1.1
    if horizon == 90:
        return 1.5 if days_left <= 15 else 1.35 if days_left <= 30 else 1.15 if days_left <= 60 else 1.0
    if horizon == 180:
        return 1.15 if days_left <= 30 else 1.30 if days_left <= 60 else 1.10 if days_left <= 120 else 1.0

    return 1.0

def coeff_ca(ca, ca_max):
    try:
        if pd.isna(ca) or ca_max is None or ca_max <= 0:
            return 1.0
        return 1 + 0.1 * (float(ca) / ca_max)
    except Exception:
        return 1.0

def compute_charge(row, ca_max, horizon):
    score = 0.5 * get_complexite(row["complexite"]) + 0.5 * get_segmentation(row["segmentation"])
    return (
        score
        * get_transfo(row["transformation"])
        * coeff_urgence(days_until(row["date_echeance"]), horizon)
        * coeff_ca(row["ca"], ca_max)
    )

# =========================================================
# EXPORT EXCEL
# =========================================================

def export_excel(df_detail, synthese_par_horizon):
    today = datetime.now().strftime("%Y-%m-%d_%Hh%M")

    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)

    filename = f"charge_cdp_{today}.xlsx"
    file_path = export_dir / filename

    # =====================================================
    # CONSTRUCTION SYNTHÈSE : 1 ligne par CDP
    # =====================================================

    # dictionnaire final
    synthese = {}

    for horizon, rows in synthese_par_horizon.items():

        for r in rows:

            cdp = r["cdp"]

            if cdp not in synthese:
                synthese[cdp] = {"CDP": cdp}

            synthese[cdp][f"Charge {horizon}"] = r["charge_cdp"]
            synthese[cdp][f"Capacité {horizon}"] = r["capacite"]
            synthese[cdp][f"Taux {horizon} %"] = r["taux_charge_%"]

    df_synthese = pd.DataFrame(list(synthese.values()))

    # tri optionnel par charge 1M
    if "Charge 1M" in df_synthese.columns:
        df_synthese = df_synthese.sort_values("Charge 1M", ascending=False)

    # =====================================================
    # EXPORT EXCEL
    # =====================================================

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:

        df_synthese.to_excel(
            writer,
            sheet_name="Synthese_CDP",
            index=False
        )

        df_detail.to_excel(
            writer,
            sheet_name="Detail_Projets",
            index=False
        )

    return filename


# =========================================================
# ENDPOINT PRINCIPAL
# =========================================================

@app.post("/process")
async def process_file(file: UploadFile = File(...)):

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    df = pd.read_excel(file_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # mapping colonnes
    col_map = {
        "cdp": ["cdp", "cdp mobilier"],
        "statut": ["statut de l'opportunité", "statut"],
        "date_echeance": ["date d'échéance du projet", "échéance"],
        "complexite": ["complexité", "complexité du projet"],
        "segmentation": ["segmentation", "segmentation mob"],
        "transformation": ["tx de transfo", "tx de transfo mob"],
        "ca": ["ca", "ca potentiel"],
    }

    def pick(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    cols = {k: pick(v) for k, v in col_map.items()}
    if any(v is None for v in cols.values()):
        raise HTTPException(400, f"Colonnes manquantes : {cols}")

    # dataframe MÉTIER UNIQUE
    work = pd.DataFrame({
        "cdp": df[cols["cdp"]],
        "statut": df[cols["statut"]],
        "date_echeance": df[cols["date_echeance"]].apply(parse_date_safe),
        "complexite": df[cols["complexite"]],
        "segmentation": df[cols["segmentation"]],
        "transformation": df[cols["transformation"]],
        "ca": df[cols["ca"]],
    })

    work["statut_norm"] = work["statut"].apply(normalize_str)
    devis = work[work["statut_norm"] == "devis en cours"].copy()

    ca_max = pd.to_numeric(devis["ca"], errors="coerce").max()
    ca_max = None if pd.isna(ca_max) else float(ca_max)

    horizons = [
        ("1M", 30, CHARGE_CONFIG.charge_max_1m),
        ("3M", 90, CHARGE_CONFIG.charge_max_3m),
        ("6M", 180, CHARGE_CONFIG.charge_max_6m),
    ]

    result = {}

    for label, days, cap in horizons:
        tmp = devis.copy()
        tmp["charge_projet"] = tmp.apply(lambda r: compute_charge(r, ca_max, days), axis=1)

        agg = tmp.groupby("cdp")["charge_projet"].sum().reset_index()
        agg["taux_charge_%"] = agg["charge_projet"].apply(lambda x: x / cap * 100)

        result[label] = [
            {
                "cdp": r["cdp"],
                "charge_cdp": float(r["charge_projet"]),
                "capacite": float(cap),
                "taux_charge_%": float(r["taux_charge_%"]),
            }

            for _, r in agg.sort_values("charge_projet", ascending=False).iterrows()
        ]

    filename = export_excel(devis.copy(), result)
    upload_to_drive(EXPORT_DIR / filename, filename)

    return sanitize_json({
        "ok": True,
        "export_file": filename,
        "nb_devis_en_cours": len(devis),
        "charge_par_horizon": result,
    })

@app.get("/download/{filename}")
def download(filename: str):
    path = EXPORT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Fichier introuvable")
    return FileResponse(path)








