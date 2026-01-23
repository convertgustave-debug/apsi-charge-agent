from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# =========================================================
# CONFIG CHARGE MAX (capacité max en points par horizon)
# =========================================================

class ChargeConfig(BaseModel):
    charge_max_1m: float = 12
    charge_max_3m: float = 25
    charge_max_6m: float = 40

CHARGE_CONFIG = ChargeConfig()


@app.get("/")
def root():
    return {"status": "ok", "message": "APSI charge agent is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/set_charge")
def set_charge(config: ChargeConfig):
    global CHARGE_CONFIG
    CHARGE_CONFIG = config
    return {"ok": True, "charge_config": CHARGE_CONFIG.model_dump()}


# =========================================================
# UTILS
# =========================================================

def sanitize_json(obj):
    """
    Convertit récursivement NaN / +Inf / -Inf en None
    pour éviter: ValueError: Out of range float values are not JSON compliant
    """
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


def parse_date_safe(value):
    """
    Essaye de convertir une valeur Excel en date.
    Renvoie un objet date ou None.
    """
    if pd.isna(value):
        return None
    try:
        d = pd.to_datetime(value, errors="coerce")
        if pd.isna(d):
            return None
        return d.date()
    except Exception:
        return None


def days_until(d: date):
    if d is None:
        return None
    return (d - date.today()).days


# =========================================================
# CALCULS MÉTIER
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
    "20%": 0.70,
    "40%": 0.85,
    "60%": 1.00,
    "80%": 1.15,
    20: 0.70,
    40: 0.85,
    60: 1.00,
    80: 1.15,
    0.2: 0.70,
    0.4: 0.85,
    0.6: 1.00,
    0.8: 1.15,
}

# Valeurs moyennes demandées si case vide
AVG_COMPLEXITE = 2.5
AVG_SEGMENTATION = 2.5
AVG_TRANSFO = (0.70 + 0.85 + 1.00 + 1.15) / 4  # 0.925


def get_complexite(val):
    if pd.isna(val):
        return AVG_COMPLEXITE
    try:
        v = float(val)
        if v < 1 or v > 4:
            return AVG_COMPLEXITE
        return v
    except Exception:
        return AVG_COMPLEXITE


def get_segmentation(val):
    if pd.isna(val):
        return AVG_SEGMENTATION
    s = normalize_str(val)
    if s is None:
        return AVG_SEGMENTATION
    return float(SEGMENTATION_MAP.get(s, AVG_SEGMENTATION))


def get_transfo_coeff(val):
    if pd.isna(val):
        return AVG_TRANSFO

    # si c'est numérique (20,40,60,80 ou 0.2 etc.)
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(TRANSFO_MAP.get(val, AVG_TRANSFO))

    s = str(val).strip()
    return float(TRANSFO_MAP.get(s, AVG_TRANSFO))


def coeff_urgence(days_left, horizon_days):
    """
    Coeff urgence basé sur la fenêtre (horizon) + nb jours restants
    Notes (selon ton schéma):
    - 1 mois (<=30j) : <=7j ->1.4 ; 8-14 ->1.25 ; 15-30 ->1.10
    - 3 mois (~90j) : <=15 ->1.5 ; 16-30 ->1.35 ; 31-60 ->1.15 ; 61-90 ->1.00
    - 6 mois (~180j): <=30 ->1.15 ; 31-60 ->1.30 ; 61-120 ->1.10 ; 121-180 ->1.00
    Si date manquante -> coeff = 1.0 (ne pas “inventer” d’urgence)
    """
    if days_left is None:
        return 1.0

    # si déjà en retard (négatif) => urgence max
    if days_left < 0:
        return 1.5

    if horizon_days <= 30:
        if days_left <= 7:
            return 1.40
        if days_left <= 14:
            return 1.25
        return 1.10

    if horizon_days <= 90:
        if days_left <= 15:
            return 1.50
        if days_left <= 30:
            return 1.35
        if days_left <= 60:
            return 1.15
        return 1.00

    # horizon 180
    if days_left <= 30:
        return 1.15
    if days_left <= 60:
        return 1.30
    if days_left <= 120:
        return 1.10
    return 1.00


def coeff_ca(ca_val, ca_max_periode):
    """
    Coeff CA = 1 + 0.1 * (CA_projet / CA_max_de_la_période)
    Si CA manquant -> 1.0
    Si ca_max_periode manquant ou 0 -> 1.0
    """
    if pd.isna(ca_val):
        return 1.0
    if ca_max_periode is None or ca_max_periode <= 0:
        return 1.0
    try:
        ca = float(ca_val)
        return 1.0 + 0.1 * (ca / float(ca_max_periode))
    except Exception:
        return 1.0


def compute_charge_projet(row, ca_max_periode, horizon_days):
    """
    Charge projet = Score * coeff_transfo * coeff_urgence * coeff_CA
    Score = 0.5*complexité + 0.5*segmentation
    """
    complexite = get_complexite(row.get("complexite"))
    segmentation = get_segmentation(row.get("segmentation"))
    score = 0.5 * complexite + 0.5 * segmentation

    transfo = get_transfo_coeff(row.get("transformation"))

    d_echeance = row.get("date_echeance")
    jours = days_until(d_echeance)

    urg = coeff_urgence(jours, horizon_days)

    ca = row.get("ca")
    ca_coeff = coeff_ca(ca, ca_max_periode)

    charge = score * transfo * urg * ca_coeff
    return float(charge)


# =========================================================
# ENDPOINT PRINCIPAL
# =========================================================

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    # 1) Sauvegarde temporaire
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 2) Lecture Excel
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture Excel: {str(e)}")

    # 3) Normalisation des colonnes
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 4) Colonnes attendues (on accepte quelques variations)
    def pick_col(possibles):
        for c in possibles:
            if c in df.columns:
                return c
        return None

    col_cdp = pick_col(["cdp mobilier", "cdp", "chef de projet", "cdp mob"])
    col_statut = pick_col(["statut de l'opportunité", "statut opportunité", "statut"])
    col_echeance = pick_col(["dat d'échéance du projet", "date d'échéance du projet", "date echeance", "échéance"])
    col_complexite = pick_col(["complexité du projet", "complexite du projet", "complexite"])
    col_segmentation = pick_col(["segmentation mob", "segmentation", "segmentation mobilier"])
    col_transfo = pick_col(["tx de transfo mob", "tx de transfo", "taux de transfo", "transformation"])
    col_ca = pick_col(["ca potentiel mobilier", "ca potentiel", "ca", "montant"])

    missing = [name for name, col in [
        ("CDP", col_cdp),
        ("Statut", col_statut),
        ("Date échéance", col_echeance),
        ("Complexité", col_complexite),
        ("Segmentation", col_segmentation),
        ("Transformation", col_transfo),
        ("CA", col_ca),
    ] if col is None]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes: {missing}. Colonnes trouvées: {list(df.columns)}"
        )

    # 5) Dataframe de travail standardisé
    work = pd.DataFrame({
        "cdp": df[col_cdp],
        "statut": df[col_statut],
        "date_echeance": df[col_echeance].apply(parse_date_safe),
        "complexite": df[col_complexite],
        "segmentation": df[col_segmentation],
        "transformation": df[col_transfo],
        "ca": df[col_ca],
    })

    # 6) Filtre : uniquement "Devis en cours"
    work["statut_norm"] = work["statut"].apply(normalize_str)
    devis_en_cours = work[work["statut_norm"] == "devis en cours"].copy()

    # Si aucun projet
    if len(devis_en_cours) == 0:
        payload = {
            "ok": True,
            "nb_devis_en_cours": 0,
            "charge_cdp": [],
            "capacites_points": {
                "1M": CHARGE_CONFIG.charge_max_1m,
                "3M": CHARGE_CONFIG.charge_max_3m,
                "6M": CHARGE_CONFIG.charge_max_6m,
            },
        }
        return sanitize_json(payload)

    # 7) CA max période (pour coeff CA)
    # On prend le max CA dans les projets "devis en cours" (période)
    ca_max_periode = pd.to_numeric(devis_en_cours["ca"], errors="coerce").max()
    if pd.isna(ca_max_periode):
        ca_max_periode = None
    else:
        ca_max_periode = float(ca_max_periode)

    # 8) Calcul charge par horizon
    horizons = [
        ("1M", 30, float(CHARGE_CONFIG.charge_max_1m)),
        ("3M", 90, float(CHARGE_CONFIG.charge_max_3m)),
        ("6M", 180, float(CHARGE_CONFIG.charge_max_6m)),
    ]

    results_by_horizon = {}

    for label, horizon_days, cap in horizons:
        tmp = devis_en_cours.copy()

        # calcul charge projet
        tmp["charge_projet"] = tmp.apply(
            lambda r: compute_charge_projet(r, ca_max_periode=ca_max_periode, horizon_days=horizon_days),
            axis=1
        )

        # agrégation par CDP
        agg = tmp.groupby("cdp", dropna=True)["charge_projet"].sum().reset_index()
        agg = agg.rename(columns={"charge_projet": "charge_cdp"})

        # taux de charge
        agg["taux_charge_%"] = agg["charge_cdp"].apply(lambda x: (x / cap) * 100 if cap > 0 else None)

        # format sortie
        out = []
        for _, row in agg.iterrows():
            out.append({
                "cdp": row["cdp"],
                "charge_cdp": float(row["charge_cdp"]),
                "taux_charge_%": float(row["taux_charge_%"]) if not pd.isna(row["taux_charge_%"]) else None,
            })

        # tri du plus chargé au moins chargé
        out.sort(key=lambda x: x["charge_cdp"], reverse=True)
        results_by_horizon[label] = out

    payload = {
        "ok": True,
        "nb_devis_en_cours": int(len(devis_en_cours)),
        "ca_max_periode": ca_max_periode,
        "capacites_points": {
            "1M": float(CHARGE_CONFIG.charge_max_1m),
            "3M": float(CHARGE_CONFIG.charge_max_3m),
            "6M": float(CHARGE_CONFIG.charge_max_6m),
        },
        "charge_par_horizon": results_by_horizon,
    }

    return sanitize_json(payload)







