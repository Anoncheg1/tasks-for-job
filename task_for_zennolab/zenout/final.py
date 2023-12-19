# ------------------------------------------------- LIBRARY ---------------------------------------------
import json
import glob
import os
import numpy as np

class MyException(Exception):
    pass

def parse_file(json_file:str):
    with open(json_file, "r", encoding="utf-8") as infile:
        myfile = json.load(infile)
    plus_point = None
    train_rectangles = [None for _ in range(7)] # 7
    digits_rectangles = {}
    for i, shape in enumerate(myfile["shapes"]):
        if shape["label"] == "+":
            plus_point = shape['points'][0]
            # - convert to int:
            plus_point = (round(plus_point[0])//200, round(plus_point[0]), round(plus_point[1]))
        else:
            # - convert to int:
            dr = shape['points']
            dr = ((round(dr[0][0]), round(dr[0][1])), (round(dr[1][0]), round(dr[1][1])))

            if "_" in shape["label"]:
                train_rectangles[dr[0][0]//200] = dr
            else:
                digits_rectangles[shape["label"]] = dr
    if not all(train_rectangles):
        raise MyException("not all train_rectangles!")

    return plus_point, train_rectangles, digits_rectangles

def get_subimage(img, i=0):
    return img[0:VERTIC, HORIZ*i:HORIZ*(i+1)]


def rectangle_parser(rec, left=0, top=0):
    "substract left and top and convert to x,y,w,h"
    rr = list(rec)
    r = sorted(rr, key=lambda x: x[0])
    x1 = r[0][0]
    y1 = r[0][1]
    x2 = r[1][0]
    y2 = r[1][1]

    w = x2-x1
    h = y2-y1
    return ((x1 - left, y1 - top, w, h))
    # return rec


def hint_parser(drs):
    "get hint coordinates on hint subimage"
    hints = []
    for x in drs.values():
        if x[0][0] < HINT_HORIZ and x[0][1] > VERTIC:
            hints.append(x)
    # assert len(hints) == 2
    if len(hints) != 2:
        return None, None
    hints = sorted(hints, key=lambda x: x[0][0])
    hintsn = np.array(hints)
    hintsn[0][0][1] = hintsn[0][0][1] - VERTIC
    hintsn[0][1][1] = hintsn[0][1][1] - VERTIC
    hintsn[1][0][1] = hintsn[1][0][1] - VERTIC
    hintsn[1][1][1] = hintsn[1][1][1] - VERTIC

    return hintsn


def get_all(main_path:str = "task1.1/mixed_train_to_the_coordinates_dataset") -> (list, list, list):
    """get id's of files in dataset
    returns:
    - img_files - pathes
    - plus_points - + label
    - train_rectangles - x_x labels
    - digits_rectangles - x labels
    - hints - sorted corrdinates of xy1, xy2 on subimage"""
    a = glob.glob(main_path + "/*.jpg")
    assert len(a) > 0
    idds = [os.path.basename(x).split(".")[0] for x in a]
    img_files = []
    plus_points = []
    train_rectangles = []
    digits_rectangles = []
    hints = []
    for idd in idds:
        json_file = main_path + f"/{idd}.json"
        try:
            plus_point, train_rectangles2, digits_rectangles2 = parse_file(json_file)
        except MyException as a:
            continue
        img_files.append(main_path + f"/{idd}.jpg")
        plus_points.append(plus_point)
        train_rectangles.append(train_rectangles2)
        digits_rectangles.append(digits_rectangles2)
        hints.append(hint_parser(digits_rectangles2))
    return img_files, plus_points, train_rectangles, digits_rectangles, hints


def diff_two_rectangles(r1, r2):
    x1,y1,w1,h1 = r1
    x2,y2,w2,h2 = r2
    y_diff = abs((y1+h1/2)-(y2+h2/2))
    x_diff = abs((x1+w1/2)-(x2+w2/2))
    return np.mean([x_diff, y_diff])


def diff_two_contours(c1, c2):
    return diff_two_rectangles(cv.boundingRect(c1), cv.boundingRect(c2))


def get_subimage_roi_xywh(img, x, y, w, h):
    "img: BGR"
    return img[y:y+h,x:x+w].copy()

def get_subimage_roi_xy(img, xy1, xy2 ):
    "img: BGR"
    x1, y1 = xy1
    x2, y2 = xy2
    return img[y1:y2,x1:x2].copy()


def hsv_to_gimp(hsv_orig):
    hsv = hsv_orig.copy()
    for i in range(3):
        if i == 0:
            ranges = [0, 180]
        else:
            ranges = [0, 100]
        cv.normalize(hsv[i], hsv[i], alpha=ranges[0], beta=ranges[1],
                     norm_type=cv.NORM_MINMAX)
    return hsv, ([0, 180], [0, 100], [0, 100])

def output_histogram(img, ranges, bins = 10):
    " usage: output_histogram(hsv, [(0,255)]*3)"
    histSize = max(bins, 2)
    for i in range(3):
        hist = cv.calcHist([img[i]], [0], None, [histSize], ranges[i],
                           accumulate=False) # list of bins with values in 0-9999999 range

        # cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        print("i", i)
        [print(np.round(k), "\t", np.round(v,2)) for k,v in zip(np.linspace(ranges[i][0],ranges[i][1], bins+1)[1:], hist)]
        print()

def contours_calc_centers(contours):
    " and sort by x"
    centers = [None] *len(contours)
    for j, c in enumerate(contours):
        # print(c)
        x,y,w,h = cv.boundingRect(c)
        centers[j] = ((x+w/2), (y+h/2))
    centers = sorted(centers, key = lambda x: x[0])
    return centers

# ------------------ local ----
HORIZ = 200 # left edge of one in 7 images
VERTIC = 200 # bottom edge of 7 images
HINT_HORIZ = 135 # left edge of hint image

def get_hint_subimage(img):
    return img[VERTIC:400, 0:HINT_HORIZ].copy()

def draw_points(img, pts:list):
    for x,y in pts:
        image = cv.circle(img, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)
    # plt.imshow(image,),plt.show()


def get_centroid(pts:np.array):
    z = np.array(pts)
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv.KMEANS_RANDOM_CENTERS
    z = np.float32(z)
    compactness,labels,centers = cv.kmeans(z,2,None,criteria,10,flags)
    big_label = int(np.median(labels))
    return centers[big_label]


def match_images_swift(img_src,img_dst, distance=0.9):
    """return points on img_dst
    bigger distance -> more points"""
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_src,None)
    kp2, des2 = sift.detectAndCompute(img_dst,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    # flann = cv.FlannBasedMatcher()
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    # count = 0
    dst_matches = []
    for j,(m,n) in enumerate(matches):
        if m.distance < distance*n.distance:
            # matchesMask[j]=[1,0]
            # count+=1
            dst_matches.append(kp2[m.trainIdx])
    dst_pts = [i.pt for i in dst_matches]

    # draw_params = dict(matchColor = (0,255,0),
    #                singlePointColor = (255,0,0),
    #                matchesMask = matchesMask,
    #                flags = cv.DrawMatchesFlags_DEFAULT)
    # img3 = cv.drawMatchesKnn(img_src,kp1,img_dst,kp2,matches,None,**draw_params)
    return dst_pts


#------------------------------------------------- MAIN -----------------------------------------------------------
import cv2 as cv
import numpy as np
# from matplotlib import pyplot as plt
from dataclasses import dataclass
from shared_image_functions import find_angle, fix_angle, get_lines_c
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def find_train(src)->list:
    """ src - big image"""
    MIN_TRAIN_AREA = 100
    # --------------- 1) BGR to HSV -------------
    hsv = cv.cvtColor(src.copy(), cv.COLOR_BGR2HSV)
    # --------------- 2) split and rotate
    imgs = [hsv[0:200, 200*j:(1+j)*200].copy() for j in range(7)]
    imgs_src = [src[0:200, 200*j:(1+j)*200].copy() for j in range(7)]

    # --------------- 3) fix orientation
    # ss = imgs_src[0]
    # # ss = ss[30:, 40:]
    # a = find_angle(imgs[0], get_lines_c)
    # print("aaaaaaaaaaaaaaaaaaaaa", a)
    # ss = fix_angle(imgs_src[0], angle=a)
    # imgs = [fix_angle(img, angle=a) for img in imgs]
    # imgs_src0 = fix_angle(imgs_src[0], angle=a)
    imgs_src0 = imgs_src[0]
    hsv = np.hstack(imgs)
    # --------------- 4) find train contours
    low_H, high_H = 112, 128
    low_S, high_S = 102, 153
    low_V, high_V = 102, 204
    mask = cv.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    # print("mask.shape", mask.shape)

    dilatation_type = cv.MORPH_RECT
    dilatation_size = 5
    element = cv.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))

    img = mask
    img = cv.dilate(img, element)
    img = cv.erode(img, element)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) > MIN_TRAIN_AREA]
    # print(len(contours))
    assert len(contours) == 7

    # ------------- 5) calc contour centers --------------
    centers = contours_calc_centers(contours) # and sort
    # ------------- 6) calc distance on grid - vertical and horizontal
    centers_single = [(c[0] - 200*j,c[1]) for j, c in enumerate(centers)]
    # print(centers_single)
    # -- x
    distvec = pdist(centers_single, metric = lambda x, y: abs(abs(x[0] - y[0])/1.5 + abs(x[1] - y[1])))
    sqf = squareform(distvec)
    np.fill_diagonal(sqf, np.inf)
    i, j = np.where(sqf==sqf.min())
    i, j = i[0], j[0]
    xdist = abs(centers_single[i][0] -  centers_single[j][0])
    # print("x", centers_single[i], centers_single[j])
    # print("closest by x", xdist)
    # -- y
    distvec = pdist(centers_single, metric = lambda x, y: abs(abs(x[0] - y[0]) + abs(x[1] - y[1])/1.5))
    sqf = squareform(distvec)
    np.fill_diagonal(sqf, np.inf)

    i, j = np.where(sqf==sqf.min())
    i, j = i[0], j[0]
    # print("y", centers_single[i], centers_single[j])
    ydist = abs(centers_single[i][1] - centers_single[j][1])
    # print("closest by y", ydist)
    # ------------- 7) get vertical and horizontal figures-numbers
    rects = []
    for c in centers_single:
        # print("c", c)
        # (|  |)
        xr = (round(c[0]-xdist/2), round(c[0]+xdist/2))
        # print("xr", xr)
        # (=)
        yr = (round(c[1]-ydist/2), round(c[1]+ydist/2))
        # print("yr", yr)
        # 1-----------\.
        # -----------2/
        cx = round(c[0])
        cy = round(c[1])
        ractx = ((0, yr[0]), (cx, yr[1])) # (1, 2)
        # |1 |
        # \./2
        racty = ((xr[0], 0), (xr[1], cy)) # (1, 2)
        # ------------ 8) cut rectangle per x and y
        # -- x
        # print("subimg", ractx[0][1],(ractx[1][1] - ractx[0][1]),
        #              ractx[0][0],(ractx[1][0] - ractx[0][0]))

        subimgx = imgs_src0[ractx[0][1]:ractx[1][1],
                     ractx[0][0]:ractx[1][0]].copy()
        # -- y
        subimgy = imgs_src0[racty[0][1]:racty[1][1],
                     racty[0][0]:racty[1][0]].copy()
        rects.append({"subimgx": subimgx, "subimgy": subimgy,
        "ractx": ractx, "racty": racty, "center": c})
        # plt.imshow(src)
        # plt.show()
        # plt.imshow(subimgx)
        # plt.show()
        # plt.imshow(subimgy)
        # plt.show()
        # break

    return rects





HINT1_AREA_MIN = 2536
HINT1_AREA_MAX = 6038
HINT2_AREA_MIN = 684
HINT2_AREA_MAX = 2096

@dataclass
class ContourStats:
    circle_center_x_min: float
    circle_center_x_max: float
    circle_center_y_min: float
    circle_center_y_max: float
    circle_radius_min: float
    circle_radius_max: float
    circle_area_min: float
    circle_area_max: float


def find_object(image, circle_stats: ContourStats, conti = None):
    """ image - BGR
    loop: 1) channels, 2) threshold 3) contours
    continue: ((i, thrs), cnt)
    used in def find_hint_images """
    dilatation_type = cv.MORPH_RECT
    dilatation_size = 1
    element = cv.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))


    contour_result = None
    # -- 1)
    r = cv.split(image.copy())
    if conti is not None:
        r = r[conti[0][0]:]

    for i, gray in enumerate(r):
        if contour_result is not None:
            break
        # -- 2)
        ra = range(0, 255, 10)
        if conti is not None:
            ra = range(conti[0][1], 255, 10)
        for thrs in ra:
            if contour_result is not None:
                break
            # -- dilation
            gray = cv.erode(gray, element)

            gray = cv.dilate(gray, element)
            gray = cv.dilate(gray, element)

            _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            contours, _ = cv.findContours(bin, cv.RETR_LIST,
                                          cv.CHAIN_APPROX_SIMPLE)
            # -- 3)
            for j, cnt in enumerate(contours):
                if contour_result is not None:
                    break
                if conti is not None and all(cnt[0][0] == conti[1][0][0]):
                    continue # TODO: sort contours and filter by x,y
                # -- features of contour
                (x,y),radius = cv.minEnclosingCircle(cnt)
                area = cv.contourArea(cnt)

                if circle_stats.circle_center_x_min < x < circle_stats.circle_center_x_max \
                   and circle_stats.circle_center_y_min < y < circle_stats.circle_center_y_max \
                   and circle_stats.circle_area_min < area < circle_stats.circle_area_max \
                   and circle_stats.circle_radius_min < radius < circle_stats.circle_radius_max:

                    contour_result = cnt
                    break
    return contour_result, (i, thrs)


def find_hint_images(hint_img):
    global HINT1_X_MAX, h1s, h2s
    img_hint1 = hint_img[:,:HINT1_X_MAX] # cut hint at right)
    cnt1, conti = find_object(img_hint1, h1s)
    if cnt1 is None:
        print(i, "find object result is None")
        return None
    # -- hint2 find --
    img_hint2 = hint_img[:,HINT1_X_MAX:] # cut hint at left

    cnt2, _ = find_object(img_hint2, h2s)
    if cnt2 is None:
        print(i, "find HINT2 is None")
        return None
    r1 = cv.boundingRect(cnt1)
    x,y,w,h = cv.boundingRect(cnt2)
    x += HINT1_X_MAX
    return r1, (x,y,w,h)

# ------------------- PREPARE HINT STATISITCS
HINT1_STATS_s = {'circle_center_x_min': 38.5, 'circle_center_x_max': 50.5, 'circle_center_y_min': 81.0, 'circle_center_y_max': 93.0, 'circle_radius_min': 28.41224479675293, 'circle_radius_max': 43.840721130371094}
HINT2_STATS_s = {'circle_center_x_min': 104.0, 'circle_center_x_max': 116.5, 'circle_center_y_min': 56.5, 'circle_center_y_max': 70.5, 'circle_radius_min': 14.764923095703125, 'circle_radius_max': 25.831281661987305}
# 'circle_center_x_max': 50.5 + 'circle_radius_max': 43.840721130371094 = 94
HINT1_X_MAX = 94

HINT1_STATS_s["circle_area_min"] = HINT1_AREA_MIN
HINT1_STATS_s["circle_area_max"] = HINT1_AREA_MAX
HINT2_STATS_s["circle_area_min"] = HINT2_AREA_MIN
HINT2_STATS_s["circle_area_max"] = HINT2_AREA_MAX

# HINT2_STATS_small = {k:((v - HINT1_X_MAX) if k.startswith("circle_center_x_") else v) for k,v in HINT2_STATS.items()}

MUL = 1.3
HINT1_STATS = {}
for k,v in HINT1_STATS_s.items():
    if k.endswith('min'):
        HINT1_STATS[k] = v/ MUL
    # elif "radius_max" in k: # max
    #     HINT1_STATS[k] = v* 1.8
    else:
        HINT1_STATS[k] = v*MUL
HINT2_STATS = {}
for k,v in HINT2_STATS_s.items():
    if '_x_' in k:
        v = v - HINT1_X_MAX
    if k.endswith('min'):
        HINT2_STATS[k] = v/ MUL
    # elif "radius_max" in k: # max
    #     HINT2_STATS[k] = v* 1.8
    else:
        HINT2_STATS[k] = v *MUL


h1s = ContourStats(**HINT1_STATS)
h2s = ContourStats(**HINT2_STATS)




def get_hint_sub(img, xywh, padding):
    x,y,w,h = xywh

    x1, y1 = (x,y)
    x2, y2 = (x+w, y+h)
    # with padding approach
    HINT_PADDING = 5

    # print("wtf", hrec1, x1, y1, x2, y2)
    yh = y+h + padding
    xw = x+w + padding
    y = y - padding
    x = x - padding
    y = y if y>=0 else 0
    x =x if x>=0 else 0
    yh = yh if yh<=200 else 200
    xw = xw if xw<=HINT_HORIZ else HINT_HORIZ
    img_h1 = img[y:yh,x:xw].copy()
    img_h1 = cv.cvtColor(img_h1, cv.COLOR_BGR2GRAY)
    return img_h1


def solve_captcha(image_path):
    " return 0-6"
    src = cv.imread(image_path)
    assert src is not None, "img could not be read"
    # ------- 2) prepare hint images
    img_hint = get_hint_subimage(src)
    # ------- find hint subimages
    hrec1, hrec2 = find_hint_images(img_hint)

    # print(hrec1, hrec2)
    # ------- extract hint subimages
    # plt.imshow(img_hint),plt.show()

    # ------- 3) prepare main image - fix orientation
    imgs = [src[0:200, 200*j:(1+j)*200].copy() for j in range(7)]
    a = find_angle(imgs[0], get_lines_c)

    imgs = [fix_angle(img, angle=a) for img in imgs]
    # plt.imshow(imgs[0]),plt.show()
    # ------- 4) find train

    ftrains = find_train(np.hstack(imgs))
    # break


    # print("train keys", ftrains[0].keys())
    # --------- 4) compare hint images with ones near train
    SIGNIFICANT_DIFFERENCE_BETWEEN_TRAINS = 10
    trains = []

    for j, ft in enumerate(ftrains):
        trains.append(ft['center'])
        # print(j, ft['center'])

    # ------- 5) find hint images on main image
    img_dst = imgs[0]


    for d, p in [(0.8, 5), (0.9,20), (0.8,1),  (0.7, -1)]:
        img_h1 = get_hint_sub(img_hint, hrec1, p)
        img_src1 = img_h1
        dst_pts1 = match_images_swift(img_src1,img_dst, distance=d)
        dst_pts1 = [x for x in dst_pts1 if x[0] < 70]
        if len(dst_pts1) > 9 or len(dst_pts1) == 1:
            break

    for d, p in [(0.7, 5), (0.7,20), (0.7, 3), (0.9,15), (0.8, 5)]:
        img_h2 = get_hint_sub(img_hint, hrec2, p)
        img_src2 = img_h2
        dst_pts2 = match_images_swift(img_src2,img_dst, distance=d)
        dst_pts2 = [x for x in dst_pts2 if x[1] < 70]
        if len(dst_pts2) > 9 or len(dst_pts2) == 1:
            break
    # plt.imshow(img_dst),plt.show()
    # print("dst_pts1", len(dst_pts1))
    # TODO: may be null

    # TODO: may be null
    # print(dst_pts1)
    # print(dst_pts2)

    # print("dst_pts1", dst_pts1)
    # print("dst_pts2", dst_pts2)
    # center1 = get_centroid(dst_pts1)
    if len(dst_pts1) == 1:
        center1 = dst_pts1[0]
    elif len(dst_pts1)> 1:
        center1 = get_centroid(dst_pts1)
    else:
        # print("error1")
        return 0

    if len(dst_pts2) == 1:
        center2 = dst_pts2[0]
    elif len(dst_pts2)> 1:
        center2 = get_centroid(dst_pts2)
    else:
        # print("error2")
        return 0

    # print("len(dst_pts1), len(dst_pts2)", len(dst_pts1), len(dst_pts2))
    # print(center1)
    # print(center2)
    # # trains = [[x[0], x[1]] for x in trains]
    # # trains = sum(trains,[])
    # print("trains", trains)
    # v = [plus_points[i][0]] + list(center1) + list(center2) + trains
    # np.train
    # assert len(v) == 19
    # table.append(v)
    # print("wtf", i,plus_points[i][0], center1, center2, trains)
    # draw_points(img_dst, [center1])
    # plt.imshow(img_h2),plt.show()
    # draw_points(img_dst.copy(), dst_pts2)
    # draw_points(img_dst, [center2])
    r = []
    for t in trains:
        # print(t, center1, center2)
        # print(abs(center1[1] - t[1]), abs(center2[0] - t[0]))
        sub = abs(center1[1] - t[1]) + abs(center2[0] - t[0])
        r.append(sub)
    return np.argmin(r) #, center1, center2

# ---------------------- TEST ON ALL ---------------
# img_files, plus_points, train_rectangles, digits_rectangles, hints = \
#   get_all(main_path = "task1.1/mixed_train_to_the_coordinates_dataset")

from typing import List
import sys

def main(args: List[str]) -> None:
    img_files, plus_points, train_rectangles, digits_rectangles, hints = get_all(args[0])
    correct = [x[0] for x in plus_points]

    predicted = []
    for i, img_file in enumerate(img_files):
        # if i > 3:
        #     break
        pred = solve_captcha(img_file)
        predicted.append(pred)
        print("predicted, true:", pred, correct[i], img_file)
    acc = sum([x==y for x, y in zip(correct, predicted)])/(len(predicted))
    print("Accuracy:", round(acc,3))

if __name__ == "__main__":
    if len(sys.argv) >1:
        main(sys.argv[1:])
    else:
        print("use:\n python final.py /home/user/mixed_train_to_the_coordinates_dataset")
