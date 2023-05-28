import pandas as pd

ff = open("new_dataset/emnist-balanced-mapping.txt")


mDict = {}

for line in ff:
    splitted = line.split(" ")
    mDict[splitted[0]] = chr(int(splitted[1].replace("\n", "")))


for x in mDict.values():
    print(f"{x}")
