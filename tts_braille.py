import pyttsx3
import uuid

engine = pyttsx3.init()


def generate_speech(text_input:str):
    fname = f"braille_speech/{str(uuid.uuid1())}.mp3"
    engine.save_to_file(text_input , fname)
    engine.runAndWait()
    return fname 
