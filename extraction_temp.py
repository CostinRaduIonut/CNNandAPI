import cv2 as cv
import random
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def group_by_line_subroutine(bb, separation_distance_min, separation_distance_max):
    separated_lines = []
    line = []
    for i in range(len(bb) - 1):
        r1 = bb[i]
        r2 = bb[i + 1]

        y1, y2 = r1[1], r2[1]
        h1, h2 = r1[3], r2[3]

        yc1 = y1 + h1 // 2
        yc2 = y2 + h2 // 2

        dy = abs(yc1 - yc2)

        if dy <= separation_distance_max + h1//2 + h2//2:
            line.append(r1)
        else:
            line.append(r1)
            separated_lines.append(line)
            line = []
    line.append(bb[-1])
    separated_lines.append(line)

    return separated_lines


def grouping_subroutine(bb, min_dist, max_dist):
    line = []
    grouped_bb = []
    for i in range(len(bb) - 1):
        r1 = bb[i]
        r2 = bb[i + 1]
        x1, x2 = r1[0], r2[0]
        w1, w2 = r1[2], r2[2]

        xc1 = x1 + w1 // 2
        xc2 = x2 + w2 // 2

        right = x1 + w1
        left = x2

        dx = abs(xc2 - xc1)
        if abs(right - left) < min_dist // 2:
            line.append(r1)
        else:
            line.append(r1)
            grouped_bb.append(line)
            line = []
    line.append(bb[-1])
    grouped_bb.append(line)

    return grouped_bb




def get_distance_vector(bb) -> list:
    distances = []
    for i in range(len(bb) - 1):
        r1 = bb[i]
        r2 = bb[i + 1]
        x1, x2 = r1[0], r2[0]
        w1, w2 = r1[2], r2[2]

        xc1 = x1 + w1 // 2
        xc2 = x2 + w2 // 2

        dx = abs(xc2 - xc1)
        distances.append(dx)
    return distances


def extract(img_path):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)

    ret,thresh = cv.threshold(blur,127,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # cv.imshow("test",thresh)
    # cv.waitKey(0)
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    bb = []
    hierarchy = hierarchy[0]
    separated_lines = []

    # cv.waitKey(0)

    # create and filter unnecessary bounding boxes
    for i, cnt in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)

        if hierarchy[i][3] == 0 and w * h > 32:
            bb.append((x, y, w, h))
            # bb.append((x, y - h // 2, w, h + h //2))


    if len(bb) == 0:
        return []

    if len(bb) == 1:
        x, y, w, h = bb[0][0], bb[0][1], bb[0][2], bb[0][3]
        W, H = w + 8, h + 8
        new_img = np.zeros((H, W), dtype=np.uint8)
        xx = (W - w) // 2
        yy = (H - h) // 2
        gray_subimg = gray[y:y+h, x: x + w]
        inverted_img = cv.bitwise_not(gray_subimg)

        new_img[yy:yy+inverted_img.shape[0], xx:xx +
                inverted_img.shape[1]] = inverted_img
        new_img = cv.resize(new_img, (28, 28))
        new_img = cv.rotate(new_img, cv.ROTATE_90_COUNTERCLOCKWISE)
        new_img = cv.flip(new_img, 0)

        return [[[new_img]]]

    separation_distance = bb[0][3]

    for b in bb:
        if b[3] > separation_distance:
            separation_distance = b[3]
    separation_distance_max = max(bb, key=lambda k: (k[1], k[0]))[3]
    separation_distance_min = min(bb, key=lambda k: (k[1], k[0]))[2]
    # sort bounding boxes
    bb.sort(key=lambda k: k[1])

    separated_lines = group_by_line_subroutine(bb, 0, 0)
    grouped_bb = []
    for separated_line in separated_lines:
        separated_line.sort(key=lambda k: k[0])
        filtered_x = separated_line[:]

        # for box in separated_line:
        #     print(box)
        #     x, y, w, h = box[0], box[1], box[2], box[3]
        #     total_pixels = w * h 
        #     subimg = gray[y : y + h, x : x + h]
        #     black_pixels = np.count_nonzero(subimg == 0)
        #     percentage = black_pixels / total_pixels
        #     if percentage < .56:
        #         filtered_x.append(box)

        if len(filtered_x) > 1:
            vector_distance = get_distance_vector(filtered_x)
            min_w = min(filtered_x, key=lambda k: k[2])[2]
            max_w = max(filtered_x, key=lambda k: k[2])[2]
            min_dist = np.min(vector_distance)
            max_dist = np.max(vector_distance) * 2
            temp = grouping_subroutine(filtered_x, min_dist, max_dist)
        else:
            temp=[filtered_x]
        grouped_bb.append(temp)

    line2 = []
    line3 = []
    grouped_subimgs = []
    for group in grouped_bb:
        line2 = []

        for subgroup in group:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            line3 = []
            # subgroup = apply_operation_on_subgroup(subgroup)

            for b in subgroup:
                x, y, w, h = b[0], b[1], b[2], b[3]
                W, H = w + 8, h + 8
                new_img = np.zeros((H, W), dtype=np.uint8)
                xx = (W - w) // 2
                yy = (H - h) // 2
                gray_subimg = gray[y:y+h, x: x + w]
                inverted_img = cv.bitwise_not(gray_subimg)

                new_img[yy:yy+inverted_img.shape[0], xx:xx +
                        inverted_img.shape[1]] = inverted_img
                new_img = cv.resize(new_img, (28, 28))
                new_img = cv.rotate(new_img, cv.ROTATE_90_COUNTERCLOCKWISE)
                new_img = cv.flip(new_img, 0)

                # cv.imshow("a", new_img)
                # cv.waitKey(0)

                cv.rectangle(img, (x, y), (x + w, y + h), color, 1)

                # resized_img = cv.resize(new_img, (28, 28))

                line3.append(new_img)
            line2.append(line3)
        grouped_subimgs.append(line2)
    
    # cv.imshow("a", img)
    # cv.waitKey(0)

    return grouped_subimgs


# extract("Untitled.jpg")
