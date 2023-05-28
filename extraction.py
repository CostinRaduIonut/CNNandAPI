import cv2 as cv
import random


# sterge

def extract(img_path):
    img = cv.imread(img_path)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(
        blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    bb = []
    new_bb = []
    subimgs = []
    hierarchy = hierarchy[0]

    # create and filter unnecessary bounding boxes
    for i, cnt in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)
        if hierarchy[i][3] == 0:
            bb.append((x, y, w, h))

    # sort bounding boxes
    bb.sort(key=lambda k: k[0])

    min_dist = 28
    max_dist = 52
    grouped_bb = []
    vector_distance = []
    grouped_vectors = []
    line = []

    for i in range(len(bb) - 1):
        r1 = bb[i]
        r2 = bb[i + 1]
        x1, x2 = r1[0], r2[0]
        w1, w2 = r1[2], r2[2]

        xc1 = x1 + w1 // 2
        xc2 = x2 + w2 // 2

        dx = abs(xc2 - xc1)
        if dx <= min_dist:
            line.append(r1)
        elif dx >= max_dist:
            line.append(r1)
            grouped_bb.append(line)
            line = []
    line.append(bb[-1])
    grouped_bb.append(line)
    # print(grouped_bb)

    # for group in grouped_bb:
    #     color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #     for rect in group:
    #         cv.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, 1)

    # cv.imshow("a", img)
    # cv.waitKey(0)

    xx = 0
    yy = 0

    grouped_subimgs = []

    for group in grouped_bb:
        line = []
        for rect in group:
            # cv.rectangle(
            #     img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, 1)
            x, y, w, h = rect[0], rect[1], rect[2], rect[3]
            inverted_img = cv.bitwise_not(gray[y-yy:y+h+yy, x-xx: x + w+xx])
            resized_img = cv.resize(inverted_img, (28, 28))
            line.append(resized_img)
        grouped_subimgs.append(line)


    print(len(grouped_subimgs))

    # for b in bb:
    #     x, y, w, h = b[0], b[1], b[2], b[3]
    #     inverted_img = cv.bitwise_not(gray[y-yy:y+h+yy, x-xx: x + w+xx])
    #     print('test', inverted_img.shape)
    #     # blurred_img = cv.GaussianBlur(inverted_img, (9, 9), 0)
    #     resized_img = cv.resize(inverted_img, (28, 28))
    #     subimgs.append(resized_img)

    return grouped_subimgs


# extract("Untitled.jpg")
