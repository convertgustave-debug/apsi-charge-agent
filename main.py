from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

# =========================================================
# APP
# =========================================================

app = FastAPI()

UPLOAD_DIR = Path("uploads")
EXPORT_DIR = Path("exports")
UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

# =========================================================
# CONFIG
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
    return str(x).strip().lower() if not pd.isna(x) else None

def parse_date_safe(x):
    try:
        d = pd.to_datetime(x, errors="coerce")
        return d.date() if not pd.isna(d) else None
    except:
        return None

def days_until(d):
    return (d - date.today()).days if d else None

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
    20: 0.70, 40: 0.85, 60: 1.00, 80: 1.15,
    "20%": 0.70, "40%": 0.85, "60%": 1.00, "80%": 1.15,
    0.2: 0.70, 0.4: 0.85, 0.6: 1.00, 0.8: 1.15,
}

AVG_SCORE = 2.5
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
    if horizon == 180:
        return 1.15 if days_left <= 30 else 1.30 if days_left <= 60 else 1.10 if days_left <= 120 else 1.00

    return 1.0

def coeff_ca(ca, ca_max):
    if pd.isna(ca) or not ca_max or ca_max <= 0:
        return 1.0
    return 1.0 + 0.1 * (float(ca) / ca_max)

def compute_charge(row, ca_max, horizon):
    complexite = row["complexite"] if pd.notna(row["complexite"]) else AVG_SCORE
    segmentation = SEGMENTATION_MAP.get(normalize_str(row["segmentation"]), AVG_SCORE)
    score = 0.5 * complexite + 0.5 * segmentation

    transfo = TRANSFO_MAP.get(row["transformation"], AVG_TRANSFO)
    urgence = coeff_urgence(days_until(row["date_echeance"]), horizon)
    ca_c = coeff_ca(row["ca"], ca_max)

    return score * transfo * urgence * ca_c

# =========================================================
# EXPORT
# =========================================================

def export_excel(df_detail, synthese):
    fname = f"charge_cdp_{date.today().isoformat()}.xlsx"
    path = EXPORT_DIR / fname

    synth_rows = []
    for h, rows in synthese.items():
        for r in rows:
            synth_rows.append({**r, "horizon": h})

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame(synth_rows).to_excel(w, "Synthese_CDP", index=False)
        df_detail.to_excel(w, "Detail_Projets", index=False)

    return fname

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
    ca_max = pd.to_numeric(devis["ca"], errors="coerce").max()

    horizons = {
        "1M": (30, CHARGE_CONFIG.charge_max_1m),
        "3M": (90, CHARGE_CONFIG.charge_max_3m),
        "6M": (180, CHARGE_CONFIG.charge_max_6m),
    }

    result = {}

    for label, (days, cap) in horizons.items():
        devis["charge"] = devis.apply(lambda r: compute_charge(r, ca_max, days), axis=1)
        agg = devis.groupby("cdp")["charge"].sum().reset_index()
        agg["taux_charge_%"] = (agg["charge"] / cap) * 100

        result[label] = sorted(
            agg.to_dict("records"),
            key=lambda x: x["charge"],
            reverse=True
        )

    filename = export_excel(df, result)

    return sanitize_json({
        "ok": True,
        "export_file": filename,
        "nb_devis_en_cours": len(devis),
        "capacites_points": horizons,
        "charge_par_horizon": result
    })

@app.get("/download/{filename}")
def download(filename: str):
    file = EXPORT_DIR / filename
    if not file.exists():
        raise HTTPException(404, "Fichier introuvable")
    return FileResponse(file)
