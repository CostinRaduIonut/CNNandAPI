def encrypt(text, key):
    encrypted_text = ""
    for char in text:
        if char.isalpha():
            ascii_offset = ord('a') if char.islower() else ord('A')
            encrypted_char = chr((ord(char) - ascii_offset + key) % 26 + ascii_offset)
            encrypted_text += encrypted_char
        elif char.isdigit() or char in ('.', ','):
            encrypted_char = chr((ord(char) + key) % 256)
            encrypted_text += encrypted_char
        else:
            encrypted_text += char
    return encrypted_text


def decrypt(encrypted_text, key):
    decrypted_text = ""
    for char in encrypted_text:
        if char.isalpha():
            ascii_offset = ord('a') if char.islower() else ord('A')
            decrypted_char = chr((ord(char) - ascii_offset - key) % 26 + ascii_offset)
            decrypted_text += decrypted_char
        elif char.isdigit() or char in ('.', ','):
            decrypted_char = chr((ord(char) - key) % 256)
            decrypted_text += decrypted_char
        else:
            decrypted_text += char
    return decrypted_text
