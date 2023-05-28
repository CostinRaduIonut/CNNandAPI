import keras
import numpy as np
import tensorflow as tf
import string
import extraction
import cv2 as cv
import uuid
import extraction_temp
from autocorrector import autocorrect
from tts_braille import generate_speech
import matplotlib
import keras.preprocessing.image as process
from PIL import Image, ImageFilter


litere = string.digits + string.ascii_uppercase + "abdefghnqrt"
classes = [x for x in litere]
number_symbol = [
    [0, 1],
    [0, 1],
    [1, 1]
]
uppercase_symbol = [
    [0, 0],
    [0, 0],
    [0, 1]
]
numbers = [
    [
        [0, 1],
        [1, 1],
        [0, 0]
    ],

    [
        [1, 0],
        [0, 0],
        [0, 0]
    ],

    [
        [1, 0],
        [1, 0],
        [0, 0]
    ],

    [
        [1, 1],
        [0, 0],
        [0, 0]
    ],

    [
        [1, 1],
        [0, 1],
        [0, 0]
    ],

    [
        [1, 0],
        [0, 1],
        [0, 0]
    ],
    [
        [1, 1],
        [1, 0],
        [0, 0]
    ],

    [
        [1, 1],
        [1, 1],
        [0, 0]
    ],

    [
        [1, 0],
        [1, 1],
        [0, 0]
    ],

    [
        [0, 1],
        [1, 0],
        [0, 0]
    ],
]

characters = {

    '.': [
        [0, 0],
        [0, 0],
        [0, 1]
    ],
    '#': [
        [0, 1],
        [0, 1],
        [1, 1]
    ],

    '0':   [
        [0, 1],
        [1, 1],
        [0, 0]
    ],

    '1':  [
        [1, 0],
        [0, 0],
        [0, 0]
    ],

    '2':  [
        [1, 0],
        [1, 0],
        [0, 0]
    ],

    '3':   [
        [1, 1],
        [0, 0],
        [0, 0]
    ],

    '4':   [
        [1, 1],
        [0, 1],
        [0, 0]
    ],

    '5':  [
        [1, 0],
        [0, 1],
        [0, 0]
    ],
    '6':  [
        [1, 1],
        [1, 0],
        [0, 0]
    ],

    '7': [
        [1, 1],
        [1, 1],
        [0, 0]
    ],

    '8':  [
        [1, 0],
        [1, 1],
        [0, 0]
    ],

    '9':  [
        [0, 1],
        [1, 0],
        [0, 0]
    ],

    'a': [
        [1, 0],
        [0, 0],
        [0, 0]
    ],

    'b':  [
        [1, 0],
        [1, 0],
        [0, 0]
    ],

    'c':   [
        [1, 1],
        [0, 0],
        [0, 0]
    ],

    'd':  [
        [1, 1],
        [0, 1],
        [0, 0]
    ],

    'e':  [
        [1, 0],
        [0, 1],
        [0, 0]
    ],
    'f':  [
        [1, 1],
        [1, 0],
        [0, 0]
    ],

    'g': [
        [1, 1],
        [1, 1],
        [0, 0]
    ],

    'h': [
        [1, 0],
        [1, 1],
        [0, 0]
    ],

    'i':  [
        [0, 1],
        [1, 0],
        [0, 0]
    ],
    'j': [
        [0, 1],
        [1, 1],
        [0, 0]
    ],
    'k': [
        [1, 0],
        [0, 0],
        [1, 0]
    ],
    'l': [
        [1, 0],
        [1, 0],
        [1, 0]
    ],
    'm': [
        [1, 1],
        [0, 0],
        [1, 0]
    ],
    'n': [
        [1, 1],
        [0, 1],
        [1, 0]
    ],
    'o': [
        [1, 0],
        [0, 1],
        [1, 0]
    ],
    'p': [
        [1, 1],
        [1, 0],
        [1, 0]
    ],
    'q': [
        [1, 1],
        [1, 1],
        [1, 0]
    ],
    'r': [
        [1, 0],
        [1, 1],
        [1, 0]
    ],
    's': [
        [0, 1],
        [1, 0],
        [1, 0]
    ],
    't': [
        [0, 1],
        [1, 1],
        [1, 0]
    ],
    'u': [
        [1, 0],
        [0, 0],
        [1, 0]
    ],
    'v': [
        [1, 0],
        [1, 0],
        [1, 1]
    ],
    'w': [
        [0, 1],
        [1, 1],
        [0, 1]
    ],
    'x': [
        [1, 1],
        [0, 0],
        [1, 1]
    ],
    'y': [
        [1, 1],
        [0, 1],
        [1, 1]
    ],
    'z': [
        [1, 0],
        [0, 1],
        [1, 1]
    ],
     'A': [
        [1, 0],
        [0, 0],
        [0, 0]
    ],

    'B':  [
        [1, 0],
        [1, 0],
        [0, 0]
    ],

    'C':   [
        [1, 1],
        [0, 0],
        [0, 0]
    ],

    'D':  [
        [1, 1],
        [0, 1],
        [0, 0]
    ],

    'E':  [
        [1, 0],
        [0, 1],
        [0, 0]
    ],
    'F':  [
        [1, 1],
        [1, 0],
        [0, 0]
    ],

    'G': [
        [1, 1],
        [1, 1],
        [0, 0]
    ],

    'H': [
        [1, 0],
        [1, 1],
        [0, 0]
    ],

    'I':  [
        [0, 1],
        [1, 0],
        [0, 0]
    ],
    'J': [
        [0, 1],
        [1, 1],
        [0, 0]
    ],
    'K': [
        [1, 0],
        [0, 0],
        [1, 0]
    ],
    'L': [
        [1, 0],
        [1, 0],
        [1, 0]
    ],
    'M': [
        [1, 1],
        [0, 0],
        [1, 0]
    ],
    'N': [
        [1, 1],
        [0, 1],
        [1, 0]
    ],
    'O': [
        [1, 0],
        [0, 1],
        [1, 0]
    ],
    'P': [
        [1, 1],
        [1, 0],
        [1, 0]
    ],
    'Q': [
        [1, 1],
        [1, 1],
        [1, 0]
    ],
    'R': [
        [1, 0],
        [1, 1],
        [1, 0]
    ],
    'S': [
        [0, 1],
        [1, 0],
        [1, 0]
    ],
    'T': [
        [0, 1],
        [1, 1],
        [1, 0]
    ],
    'U': [
        [1, 0],
        [0, 0],
        [1, 0]
    ],
    'V': [
        [1, 0],
        [1, 0],
        [1, 1]
    ],
    'W': [
        [0, 1],
        [1, 1],
        [0, 1]
    ],
    'X': [
        [1, 1],
        [0, 0],
        [1, 1]
    ],
    'Y': [
        [1, 1],
        [0, 1],
        [1, 1]
    ],
    'Z': [
        [1, 0],
        [0, 1],
        [1, 1]
    ],
}

chars_braille = characters

canShowBraille = False

# model = keras.models.load_model("nn_letters_to_braille_updated")
model = keras.models.load_model("nn_digits_and_letters", compile=False)


def detect_subimage(image):
    img_tensor = image[:]
    # img_tensor = tf.keras.preprocessing.image.img_to_array(image)
    img_tensor = tf.expand_dims(img_tensor, -1)
    img_tensor = tf.expand_dims(img_tensor, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor /= 255.0
    # img_tensor = tf.image.resize(img_tensor, (28, 28))
    prediction = model.predict(img_tensor)
    prediction = prediction[0]
    return classes[np.argmax(prediction)], np.max(prediction)


def draw_braille(img, xpos, ypos, pattern):
    dist = 8
    radius = 2
    for i in range(len(pattern)):
        for j in range(len(pattern[0])):
            color = (255, 255, 255)
            if pattern[i][j] == 1:

                color = (0, 0, 0)

            cv.circle(img, (xpos + j * dist, ypos + i * dist),
                      radius, color, thickness=2)


# def detect_image(image_path):
#     grouped_subimgs = extraction.extract(image_path)
#     # grouped_subimgs = extraction.extract(image_path)
#     # grouped_subimgs = extraction_temp.extract(image_path)
#     text = ""
#     for group in grouped_subimgs:
#         for subimg in group:
#             c, p = detect_subimage(subimg)
#             text += c
#         text += " "


#     braille_width = 16
#     braille_height = 32
#     padding = 16
#     image_width = len(text) * 2 * braille_width + padding
#     image_height = len(text) * braille_height + padding

#     img_braille = 255 * np.ones((image_height, image_width, 1))

#     for i, letter in enumerate(text):
#         if letter.isalpha():
#             draw_braille(img_braille, padding + i * braille_width *
#                          2, padding + padding, characters[letter])

#     ffname = "braille_detectat/" + str(uuid.uuid4()) + ".jpg"
#     cv.imwrite(ffname, img_braille)

#     return text, ffname.split("/")[1].strip()


def string_to_braille(text: str, check_spelling=False, is_audio = False):
    # tts_braille.generate_speech(text)
    if check_spelling:
        text = autocorrect(text)
    # print("TEXT:", text)
    image_width = 512
    image_height = 512
    braille_width = 16
    braille_height = 32
    px = 16
    py = 16
    imgs_braille = []
    img_braille = 255 * np.ones((image_height, image_width, 1))
    y = py * 2
    right = px // 2
    print(f"is_audio = {is_audio}")
    if not is_audio:
        text = format_text(text).lower()
    for i, letter in enumerate(text):
        if right < image_width - px * 4:
            right += braille_width + px // 2
        else:
            if y < image_height - py * 4:
                y += braille_height + py // 2
            else:
                y = py * 2
                # ffname = "braille_detectat/" + str(uuid.uuid4()) + ".jpg"
                # cv.imwrite(ffname, img_braille)
                imgs_braille.append(img_braille)
                img_braille = 255 * np.ones((image_height, image_width, 1))

            right = px * 2

        if letter in characters.keys():
            draw_braille(img_braille, right, y, characters[letter])
        else:
            right += braille_width - 4

    if len(imgs_braille) > 1:
        merged_img = np.vstack(imgs_braille)
    else:
        merged_img = img_braille
    ffname = "braille_detectat/" + str(uuid.uuid4()) + ".jpg"
    cv.imwrite(ffname, merged_img)

    return text, ffname

def format_text(text: str):
    formatted_tokens = []
    positions = []
    i = 0
    tokens = text.split(" ")
    for token in tokens:
        subpositions = []
        once = False
        once0 = False
        new_token = ""
        for i, t in enumerate(token):
            if t.isnumeric() and not once:
                once = True
                once0 = False
                new_token += "#"

            if t.isalpha() and t.isupper() and not once0:
                once = False
                once0 = True
                new_token += "."
            new_token += t
        formatted_tokens.append(new_token)
    return " ".join(formatted_tokens)


def text_to_braille(image_path, can_correct=False, is_audio = False):
    image_width = 512
    image_height = 512
    braille_width = 16
    braille_height = 32
    px = 16
    py = 16
    groups = extraction_temp.extract(image_path)
    text = ""
    for group in groups:
        for subgroup in group:
            for img in subgroup:
                c, p = detect_subimage(img)
                
            
                text += c
            text += " "
    
    
    if can_correct:
        # print("CAN CHECK SPELL ")
        text = autocorrect(text)
    text = text.lower()
    text = text.rstrip()
    text = text.lstrip()
    # print(text)
    tokens = text.split(" ")
    formatted_tokens = []
    positions = []
    i = 0
    if not is_audio:
        for token in tokens:
            subpositions = []
            once = False
            new_token = ""
            for i, t in enumerate(token):
                if t.isnumeric() and not once:
                    once = True
                    new_token += "#"

                if t.isalpha():
                    once = False
                new_token += t
            formatted_tokens.append(new_token)

        text = " ".join(formatted_tokens)

    generate_speech(text)
    imgs_braille = []
    img_braille = 255 * np.ones((image_height, image_width, 1))
    y = py * 2
    right = px // 2
    for i, letter in enumerate(text):
        if right < image_width - px * 4:
            right += braille_width + px // 2
        else:
            if y < image_height - py * 4:
                y += braille_height + py // 2
            else:
                y = py * 2
                # ffname = "braille_detectat/" + str(uuid.uuid4()) + ".jpg"
                # cv.imwrite(ffname, img_braille)
                imgs_braille.append(img_braille)
                img_braille = 255 * np.ones((image_height, image_width, 1))

            right = px * 2

        if letter.isalpha() or letter.isdigit() or letter == '#':
            draw_braille(img_braille, right, y, characters[letter])
        else:
            right += braille_width - 4

    if len(imgs_braille) > 1:
        merged_img = np.vstack(imgs_braille)
    else:
        merged_img = img_braille
    ffname = "braille_detectat/" + str(uuid.uuid4()) + ".jpg"
    cv.imwrite(ffname, merged_img)

    return text, ffname.split("/")[1].strip(), merged_img


# text_to_braille("Untitled.jpg")

# print(format_text("hello world MY name is COSMOS 12345 COSMOS123456WOW"))