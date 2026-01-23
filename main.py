from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import pandas as pd
import numpy as np
import io
from datetime import datetime
import numpy as np

app = FastAPI()

# -----------------------------
# Paramètres & mappings
# -----------------------------
STATUS_COL = "statut de l'opportunité"
CDP_COL = "cdp mobilier"
DEADLINE_COL = "date d'échéance du projet"
CA_COL = "ca potentiel mob"
TRANSFO_COL = "tx de transfo mob"
COMPLEX_COL = "complexité du projet"
SEG_COL = "segmenation mob"  # (oui il y a une typo dans l'export)

# Segmentation -> score
SEG_MAP = {
    "stratégique": 4,
    "strategique": 4,
    "projet": 3,
    "réassort": 2,
    "reassort": 2,
    "à développer": 1,
    "a développer": 1,
    "a developper": 1,
}

# Transfo -> coeff
TRANSFO_COEFF = {
    20: 0.70,
    40: 0.85,
    60: 1.00,
    80: 1.15,
}


def norm_colname(c: str) -> str:
    return str(c).strip().lower()


def urgency_coeff(days_to_deadline: int, horizon: str) -> float:
    """
    Coeff urgence selon horizon (1M / 3M / 6M).
    Plus la deadline est proche, plus le coeff est élevé.
    """
    if days_to_deadline is None or np.isnan(days_to_deadline):
        return 1.0

    d = int(days_to_deadline)

    if horizon == "1M":
        if d <= 7:
            return 1.40
        if d <= 14:
            return 1.25
        if d <= 30:
            return 1.10
        return 1.0

    if horizon == "3M":
        if d <= 15:
            return 1.50
        if d <= 30:
            return 1.35
        if d <= 60:
            return 1.15
        if d <= 90:
            return 1.00
        return 1.0

    if horizon == "6M":
        if d <= 30:
            return 1.45
        if d <= 60:
            return 1.30
        if d <= 120:
            return 1.10
        if d <= 180:
            return 1.00
        return 1.0

    return 1.0


def parse_transfo_to_coeff(v) -> float:
    """
    Supporte "20%", "40", 60.0 etc.
    """
    s = str(v).replace("%", "").replace(",", ".").strip()
    try:
        val = int(float(s))
    except Exception:
        return 1.0
    return TRANSFO_COEFF.get(val, 1.0)


def compute_charge_points(df: pd.DataFrame, horizon: str, days_limit: int) -> pd.DataFrame:
    """
    Calcule la charge projet pour chaque ligne dans la fenêtre [0..days_limit]
    Renvoie un df filtré et enrichi.
    """
    today = pd.Timestamp(datetime.now().date())

    dfh = df.copy()
    dfh["days_to_deadline"] = (dfh[DEADLINE_COL] - today).dt.days

    # Garde uniquement les projets dans l'horizon
    dfh = dfh[
        (dfh["days_to_deadline"].notna())
        & (dfh["days_to_deadline"] >= 0)
        & (dfh["days_to_deadline"] <= days_limit)
    ].copy()

    # Coeff CA (bonus max +10%)
    ca_max = dfh["ca"].max() if len(dfh) else 0.0
    if ca_max <= 0:
        dfh["ca_coeff"] = 1.0
    else:
        dfh["ca_coeff"] = 1.0 + 0.10 * np.minimum(dfh["ca"] / ca_max, 1.0)

    dfh["urgence_coeff"] = dfh["days_to_deadline"].apply(lambda d: urgency_coeff(d, horizon))

    # Score de base = moyenne Complexité & Segmentation
    dfh["core_score"] = 0.5 * dfh["complexite"] + 0.5 * dfh["seg_score"]

    # Charge projet finale
    dfh["charge_projet"] = (
        dfh["core_score"] * dfh["transfo_coeff"] * dfh["urgence_coeff"] * dfh["ca_coeff"]
    )

    return dfh


def summarize_by_cdp(dfh: pd.DataFrame, capacity_points: float) -> pd.DataFrame:
    """
    Agrège par CDP Mobilier : nb projets, CA cumulé, charge points, taux charge normalisé.
    """
    if len(dfh) == 0:
        return pd.DataFrame(columns=[
            "cdp_mobilier",
            "nb_projets",
            "ca_cumule",
            "charge_points",
            "taux_charge_normalise_pct",
            "statut_charge",
        ])

    grp = (
        dfh.groupby(CDP_COL, dropna=False)
        .agg(
            nb_projets=("charge_projet", "count"),
            ca_cumule=("ca", "sum"),
            charge_points=("charge_projet", "sum"),
        )
        .reset_index()
        .rename(columns={CDP_COL: "cdp_mobilier"})
    )

    if capacity_points <= 0:
        grp["taux_charge_normalise_pct"] = np.nan
    else:
        grp["taux_charge_normalise_pct"] = 100.0 * grp["charge_points"] / float(capacity_points)

    def status_label(x: float) -> str:
        if pd.isna(x):
            return ""
        if x < 90:
            return "OK"
        if x <= 110:
            return "Attention"
        return "Surcharge"

    grp["statut_charge"] = grp["taux_charge_normalise_pct"].apply(status_label)
    grp = grp.sort_values("taux_charge_normalise_pct", ascending=False)

    return grp


# -----------------------------
# API
# -----------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    capacity_1m: float = Form(12),
    capacity_3m: float = Form(25),
    capacity_6m: float = Form(40),
):
    """
    Upload Excel -> filtre devis en cours -> calcule charge par CDP (1M/3M/6M) -> renvoie JSON.
    """

    content = await file.read()

    # Lecture Excel
    try:
        df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture Excel: {str(e)}")

    # Normalisation colonnes
    df.columns = [norm_colname(c) for c in df.columns]

    # Vérifications colonnes indispensables
    required_cols = [STATUS_COL, CDP_COL, DEADLINE_COL, CA_COL, TRANSFO_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes: {missing}. Colonnes trouvées: {list(df.columns)}",
        )

    # Filtre devis en cours
    df[STATUS_COL] = df[STATUS_COL].astype(str).str.strip().str.lower()
    df = df[df[STATUS_COL] == "devis en cours"].copy()

    # Parsing date échéance
    df[DEADLINE_COL] = pd.to_datetime(df[DEADLINE_COL], dayfirst=True, errors="coerce")

    # CA
    df["ca"] = pd.to_numeric(df[CA_COL], errors="coerce").fillna(0.0)

    # Complexité (défaut 2 si absent)
    if COMPLEX_COL in df.columns:
        df["complexite"] = pd.to_numeric(df[COMPLEX_COL], errors="coerce").fillna(2.0)
    else:
        df["complexite"] = 2.0

    # Segmentation (défaut 2 si absent)
    if SEG_COL in df.columns:
        seg = df[SEG_COL].astype(str).str.strip().str.lower()
        df["seg_score"] = seg.map(SEG_MAP).fillna(2.0)
    else:
        df["seg_score"] = 2.0

    # Transfo coeff
    df["transfo_coeff"] = df[TRANSFO_COL].apply(parse_transfo_to_coeff)

    # Infos globales
    nb_devis = int(len(df))
    ca_total = float(df["ca"].sum())

    # Horizons
    horizons = [
        ("1M", 30, capacity_1m),
        ("3M", 90, capacity_3m),
        ("6M", 180, capacity_6m),
    ]

    results = {}
    for label, days_limit, cap in horizons:
        dfh = compute_charge_points(df, horizon=label, days_limit=days_limit)
        summary = summarize_by_cdp(dfh, capacity_points=cap)
        results[label] = summary.to_dict(orient="records")


        # =========================
        # JSON SAFE (anti NaN / Inf)
        # =========================
   

    def sanitize_json(obj):
        """
        Convertit tous les NaN / +Inf / -Inf en None, récursivement,
        pour éviter: ValueError: Out of range float values are not JSON compliant
        """
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_json(v) for v in obj]
        return obj

    # Nettoyage du dataframe (au cas où)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)

    # Toujours définir charge_cdp quoi qu'il arrive
    try:
        charge_cdp
    except NameError:
        charge_cdp = None


    payload = {
        "ok": True,
        "nb_devis_en_cours": int(nb_devis),
        "ca_total_devis_en_cours": ca_total,
        "capacites_points": {
            "1M": capacity_1m,
            "3M": capacity_3m,
            "6M": capacity_6m,
        },
        "resultats_par_horizon": results,
        "charge_cdp": charge_cdp,  
}

    return sanitize_json(payload)

  






