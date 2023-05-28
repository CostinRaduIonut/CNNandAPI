from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import cv2 as cv
from detection import text_to_braille, string_to_braille
from crypting import encrypt, decrypt
import os
from pydantic import BaseModel
import uuid
from tts_braille import generate_speech
import base64
import io
import numpy as np
from PIL import Image
from pymongo import MongoClient


rootdir = "images/"
key = 5
app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("parola.txt", "r") as fisier:
    parola = fisier.read().strip()

client = MongoClient(
    "mongodb+srv://DiplomaStorage:{}@diplomastorage.fk71z3w.mongodb.net/?retryWrites=true&w=majority".format(parola))
db = client["DiplomaStorage"]
collection = db["User"]
collectionData = db["KeptData"]



class Ip(BaseModel):
    ipAddress: str


class Del(BaseModel):
    data: str


@app.post("/text-to-braille")
def get_braille(response: Response, file: UploadFile = File(...), check_spelling: bool = Form(...), file_type=Form(...)):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    cv.imwrite(f"{rootdir}{file.filename}", img)

    print(f"{rootdir}{file.filename}")

    text, fname, merged_img = text_to_braille(
        f"{rootdir}{file.filename}", check_spelling, file_type == "sound")
    print(f"TEXT : {text}")
    fname_voice = generate_speech(text)
    if file_type == "image":
        return FileResponse("braille_detectat/" + fname, media_type="application/octet-stream", filename=fname)
    elif file_type == "sound":
        return FileResponse(fname_voice, headers={f"Content-Disposition": "attachment; filename={fname_voice}"})
    else:
        response.status_code = 400
        return {"error": "Invalid file_type"}


@app.post('/string-to-braille')
async def upload_string(response: Response, text: str = Form(...), file_type=Form(...), check_spelling: bool = Form(...)):
    text, fname = string_to_braille(text, check_spelling, file_type == "sound")
    fname_voice = generate_speech(text)
    if file_type == "image":
        return FileResponse(fname, headers={"Content-Disposition": f"attachment; filename={fname}"})
    elif file_type == "sound":
        return FileResponse(fname_voice, headers={f"Content-Disposition": "attachment; filename={fname_voice}"})
    else:
        response.status_code = 400
        return {"error": "Invalid file_type"}


@app.post("/new_user")
def insert_ip(ipAddress: Ip):
    ip = encrypt(ipAddress.ipAddress, key)

    item_data = {
        "ip": ip
    }
    if collection.count_documents(item_data) > 0:
        unique_count = len(collection.distinct("ip"))
        return {"message": "Adresa IP există deja în baza de date.", "unique_count": unique_count}
    else:
        collection.insert_one(item_data)
        unique_count = len(collection.distinct("ip"))

    return {"message": "Adresa IP a fost creată", "unique_count": unique_count}


@app.post("/get_kept_string")
async def get_kept_string(ipAddress: Ip):
    ip= encrypt(ipAddress.ipAddress, key)
    results = collectionData.find({"ip":ip }, {"data": 1})
    data = [entry["data"] for entry in results]
    return {"data": data}


@app.post("/put_kept_string")
async def put_kept_string(ip: str = Form(...), data: str = Form(...)):
    ip1= encrypt(ip, key)
    data_entry = {"ip": ip1, "data": data}
    collectionData.insert_one(data_entry)
    return {"message": "Datele au fost introduse în bază de date."}


@app.delete("/delete_one")
def delete_one(data: Del):
    # Șterge înregistrarea cu data specificată
    result = collectionData.delete_one({"data": data.data})

    if result.deleted_count > 0:
        return {"message": "Înregistrarea a fost ștearsă cu succes."}
    else:
        return {"message": "Nu s-a găsit nicio înregistrare cu data specificată."}


@app.delete("/delete_all")
def delete_all(ipAddress: Ip):
    ip=encrypt(ipAddress.ipAddress, key)
    # Șterge înregistrarea cu data specificată
    result = collectionData.delete_many({"ip": ip})

    if result.deleted_count > 0:
        return {"message": "Înregistrarea a fost ștearsă cu succes."}
    else:
        return {"message": "Nu s-a găsit nicio înregistrare cu data specificată."}
