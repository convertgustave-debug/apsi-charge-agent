from fastapi import FastAPI
from pydantic import BaseModel

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

from fastapi import UploadFile, File
import os
from pathlib import Path

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




