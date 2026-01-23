from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class Payload(BaseModel):
    charge_max: int


@app.get("/")
def root():
    return {"status": "ok", "message": "APSI charge agent is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/set_charge")
def set_charge(payload: Payload):
    if payload.charge_max < 0:
        return {"ok": False, "error": "charge_max must be >= 0"}

    return {"ok": True, "charge_max": payload.charge_max}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "ok": True,
        "filename": file.filename,
        "saved_as": str(file_path),
    }


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

    # 3) Vérif colonne "nom"
    df.columns = [str(c).strip().lower() for c in df.columns]
  # Colonnes possibles pour le "nom"
possible_cols = ["nom", "nom de l'opportunité", "nom opportunité", "opportunité"]

col_found = None
for c in possible_cols:
    if c in df.columns:
        col_found = c
        break

if col_found is None:
    raise HTTPException(
        status_code=400,
        detail=f"Aucune colonne nom trouvée. Colonnes trouvées: {list(df.columns)}",
    )

noms = (
    df[col_found]
    .dropna()
    .astype(str)
    .str.strip()
    .replace("", pd.NA)
    .dropna()
    .unique()
    .tolist()
)
return {
        "ok": True,
        "nb_lignes": int(len(df)),
        "nb_noms_uniques": int(len(noms)),
        "noms": noms,
}









