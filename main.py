from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import date, datetime
import pandas as pd
import numpy as np

# =========================================================
# APP & DOSSIERS
# =========================================================

app = FastAPI()

UPLOAD_DIR = Path("uploads")
EXPORT_DIR = Path("exports")

UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

# =========================================================
# CONFIG CAPACITÉS (VALIDÉE)
# =========================================================

class ChargeConfig(BaseModel):
    charge_max_1m: float = 60
    charge_max_3m: float = 130
    charge_max_6m: float = 210

CHARGE_CONFIG = ChargeConfig()

# =========================================================
# UTILS
# =========================================================

def normalize_str(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()

def parse_date_safe(x):
    try:
        d = pd.to_datetime(x, errors="coerce")
        return d.date() if not pd.isna(d) else None
    except:
        return None

def days_until(d):
    if d is None:
        return None
    return (d - date.today()).days

# =========================================================
# RÈGLES MÉTIER (STRICTEMENT CELLES VALIDÉES)
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
    20: 0.70,
    40: 0.85,
    60: 1.00,
    80: 1.15,
    0.2: 0.70,
    0.4: 0.85,
    0.6: 1.00,
    0.8: 1.15,
    "20%": 0.70,
    "40%": 0.85,
    "60%": 1.00,
    "80%": 1.15,
}

AVG_COMPLEXITE = 2.5
AVG_SEGMENTATION = 2.5
AVG_TRANSFO = 0.925

def get_complexite(v):
    try:
        v = float(v)
        if 1 <= v <= 4:
            return v
        return AVG_COMPLEXITE
    except:
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
        if days_left <= 7:
            return 1.4
        elif days_left <= 14:
            return 1.25
        else:
            return 1.1

    if horizon == 90:
        if days_left <= 15:
            return 1.5
        elif days_left <= 30:
            return 1.35
        elif days_left <= 60:
            return 1.15
        else:
            return 1.0

    if horizon == 180:
        if days_left <= 30:
            return 1.15
        elif days_left <= 60:
            return 1.30
        elif days_left <= 120:
            return 1.10
        else:
            return 1.0

    return 1.0

def coeff_ca(ca, ca_max):

    try:

        if pd.isna(ca):
            return 1.0

        if ca_max is None or ca_max <= 0:
            return 1.0

        return 1 + 0.1 * (float(ca) / ca_max)

    except:
        return 1.0

def compute_charge(row, ca_max, horizon):

    score = (
        0.5 * get_complexite(row["complexite"])
        + 0.5 * get_segmentation(row["segmentation"])
    )

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

    now = datetime.now().strftime("%Y-%m-%d_%Hh%M")

    filename = f"charge_cdp_{now}.xlsx"

    file_path = EXPORT_DIR / filename

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

    if "Charge 1M" in df_synthese.columns:
        df_synthese = df_synthese.sort_values("Charge 1M", ascending=False)

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

    return file_path

# =========================================================
# ENDPOINT PRINCIPAL (OPTIMISÉ POUR MAKE)
# =========================================================

@app.post("/process")
async def process_file(file: UploadFile = File(...)):

    try:

        input_path = UPLOAD_DIR / file.filename

        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())

        df = pd.read_excel(input_path)

        df.columns = [c.strip().lower() for c in df.columns]

        col_map = {
            "cdp": ["cdp", "cdp mobilier"],
            "statut": ["statut de l'opportunité", "statut"],
            "date_echeance": ["date d'échéance du projet", "échéance"],
            "complexite": ["complexité", "complexité du projet"],
            "segmentation": ["segmentation", "segmentation mob"],
            "transformation": ["tx de transfo", "tx de transfo mob"],
            "ca": ["ca", "ca potentiel", "CA", "ca potentiel MOB", "CA potentiel MOB"],
        }

        def pick(cols):

            for c in cols:

                if c in df.columns:
                    return c

            return None

        cols = {k: pick(v) for k, v in col_map.items()}

        if any(v is None for v in cols.values()):
            raise HTTPException(400, f"Colonnes manquantes : {cols}")

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

        devis = work[
            work["statut_norm"] == "devis en cours"
        ].copy()

        ca_max = pd.to_numeric(
            devis["ca"],
            errors="coerce"
        ).max()

        ca_max = None if pd.isna(ca_max) else float(ca_max)

        horizons = [

            ("1M", 30, CHARGE_CONFIG.charge_max_1m),
            ("3M", 90, CHARGE_CONFIG.charge_max_3m),
            ("6M", 180, CHARGE_CONFIG.charge_max_6m),

        ]

        result = {}

        for label, days, cap in horizons:

            tmp = devis.copy()

            tmp["charge_projet"] = tmp.apply(
                lambda r: compute_charge(r, ca_max, days),
                axis=1
            )

            agg = (
                tmp.groupby("cdp")["charge_projet"]
                .sum()
                .reset_index()
            )

            agg["taux_charge_%"] = (
                agg["charge_projet"] / cap * 100
            )

            result[label] = [

                {
                    "cdp": r["cdp"],
                    "charge_cdp": round(float(r["charge_projet"]), 1),
                    "capacite": round(float(cap), 1),
                    "taux_charge_%": round(float(r["taux_charge_%"]), 1),
                }
                
                for _, r in agg.sort_values(
                    "charge_projet",
                    ascending=False
                ).iterrows()

            ]

        output_path = export_excel(devis, result)

        return FileResponse(
            path=output_path,
            filename=output_path.name,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


