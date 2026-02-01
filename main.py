from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

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
    d = pd.to_datetime(x, errors="coerce")
    return None if pd.isna(d) else d.date()

def days_until(d):
    return None if d is None else (d - date.today()).days

# =========================================================
# MÉTIER
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
    "20%": 0.70, 20: 0.70, 0.2: 0.70,
    "40%": 0.85, 40: 0.85, 0.4: 0.85,
    "60%": 1.00, 60: 1.00, 0.6: 1.00,
    "80%": 1.15, 80: 1.15, 0.8: 1.15,
}

AVG = 2.5
AVG_TRANSFO = 0.925

def coeff_urgence(days_left, horizon):
    if days_left is None:
        return 1.0
    if days_left < 0:
        return 1.5

    if horizon == 30:
        return 1.40 if days_left <= 7 else 1.25 if days_left <= 14 else 1.10
    if horizon == 90:
        return 1.50 if days_left <= 15 else 1.35 if days_left <= 30 else 1.15 if days_left <= 60 else 1.00
    return 1.15 if days_left <= 30 else 1.30 if days_left <= 60 else 1.10 if days_left <= 120 else 1.00

def compute_charge(row, ca_max, horizon):
    complexite = row["complexite"] if 1 <= row["complexite"] <= 4 else AVG
    segmentation = SEGMENTATION_MAP.get(normalize_str(row["segmentation"]), AVG)
    score = 0.5 * complexite + 0.5 * segmentation

    transfo = TRANSFO_MAP.get(row["transformation"], AVG_TRANSFO)
    urgence = coeff_urgence(days_until(row["date_echeance"]), horizon)

    ca_coeff = 1.0 if pd.isna(row["ca"]) or not ca_max else 1 + 0.1 * (row["ca"] / ca_max)

    return score * transfo * urgence * ca_coeff

# =========================================================
# EXPORT
# =========================================================

def export_excel(df_detail, synthese):
    filename = f"charge_cdp_{date.today().isoformat()}.xlsx"
    path = EXPORT_DIR / filename

    synthese_rows = []
    for horizon, rows in synthese.items():
        for r in rows:
            synthese_rows.append({"horizon": horizon, **r})

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(synthese_rows).to_excel(writer, sheet_name="Synthese", index=False)
        df_detail.to_excel(writer, sheet_name="Detail_Projets", index=False)

    return filename

# =========================================================
# ENDPOINT
# =========================================================

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    path = UPLOAD_DIR / file.filename
    with open(path, "wb") as f:
        f.write(await file.read())

    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]

    work = pd.DataFrame({
        "cdp": df["cdp"],
        "statut": df["statut"],
        "date_echeance": df["date echeance"].apply(parse_date_safe),
        "complexite": df["complexite"],
        "segmentation": df["segmentation"],
        "transformation": df["transformation"],
        "ca": df["ca"],
    })

    devis = work[work["statut"].apply(normalize_str) == "devis en cours"].copy()
    if devis.empty:
        return {"ok": True, "nb_devis_en_cours": 0}

    ca_max = devis["ca"].max()

    horizons = {
        "1M": (30, CHARGE_CONFIG.charge_max_1m),
        "3M": (90, CHARGE_CONFIG.charge_max_3m),
        "6M": (180, CHARGE_CONFIG.charge_max_6m),
    }

    charge_par_horizon = {}

    for label, (days, cap) in horizons.items():
        devis["charge"] = devis.apply(lambda r: compute_charge(r, ca_max, days), axis=1)
        agg = devis.groupby("cdp")["charge"].sum().reset_index()
        agg["taux_charge_%"] = agg["charge"] / cap * 100

        charge_par_horizon[label] = agg.sort_values("charge", ascending=False).to_dict("records")

    filename = export_excel(df, charge_par_horizon)

    return sanitize_json({
        "ok": True,
        "export_file": filename,
        "nb_devis_en_cours": len(devis),
        "charge_par_horizon": charge_par_horizon
    })

@app.get("/download/{filename}")
def download_file(filename: str):
    path = EXPORT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Fichier introuvable")
    return FileResponse(path, filename=filename)














