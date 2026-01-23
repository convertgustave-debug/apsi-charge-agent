from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
from pathlib import Path
import io
import pandas as pd

ANCHOR_COL = "Nom de l'opportunité"

def read_export_excel(content_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(content_bytes), sheet_name=0, header=None)

    header_row = None
    for i in range(min(50, len(raw))):
        row_values = raw.iloc[i].astype(str)
        if (row_values == ANCHOR_COL).any():
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"Impossible de détecter l'en-tête (colonne repère '{ANCHOR_COL}')")

    df = pd.read_excel(io.BytesIO(content_bytes), sheet_name=0, header=header_row)
    return df

app = FastAPI()


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
    # Ici on valide juste et on renvoie la valeur.
    # Plus tard, on fera le vrai calcul + stockage si tu veux.
    if payload.charge_max < 0:
        return {"ok": False, "error": "charge_max must be >= 0"}

    return {"ok": True, "charge_max": payload.charge_max}


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # On enregistre le fichier dans /uploads
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "ok": True,
        "filename": file.filename,
        "saved_as": str(file_path)
    }






