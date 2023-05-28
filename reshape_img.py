import numpy as np
import cv2 as cv 
import os 
import string 

# sterge


classes = [x for x in (string.digits + string.ascii_lowercase)]
rootdir = "dataset/"

# global rootdir
#     full_path = rootdir + subfolder + class_folder
#     image_files = os.listdir(full_path)

#     for file in image_files:
#         img = cv.imread(file)
#         img = cv.resize(img, size, interpolation= cv.INTER_LINEAR)
#         cv.imwrite(full_path, img)


# size = (xs, ys)
def resize_images(subfolder,size):
    global rootdir
    full_path = rootdir + subfolder
    class_dirs = os.listdir(full_path)

    for class_dir in class_dirs:
        image_files = os.listdir(full_path + "/" + class_dir)
        for file in image_files:
            img = cv.imread(full_path + "/" + class_dir + "/" + file)
            img = cv.resize(img, size, interpolation = cv.INTER_LINEAR)
            cv.imwrite(full_path + "/" + class_dir + "/" + file, img)


# resize_images("training_data", (32, 32))
# resize_images("testing_data", (32, 32))