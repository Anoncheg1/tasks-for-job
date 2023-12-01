import cv2 as cv
import numpy as np
import math
# from PIL import Image
from typing import Callable


def img_to_small(img, height_target=575):  # TODO: resize by smallest dimension
    scale_percent = round(height_target / img.shape[1], 3)
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    img_resized = cv.resize(img, dim)
    return img_resized, scale_percent


def scale_rect(rect, img_orig_shape, img_small_shape, edge, scale_percent) -> tuple:
    xs = img_orig_shape[0] / img_small_shape[0]  # scale ration x
    ys = img_orig_shape[1] / img_small_shape[1]  # scale ration y

    center, size, theta = rect

    center = (round((center[0] + edge) * xs), round((center[1] + edge) * ys))
    size = (int(size[0] / scale_percent), int(size[1] / scale_percent))
    rect = (center, size, theta)

    return rect


def scale_box(box, img_orig_shape, img_small_shape, edge) -> tuple:
    xs = img_orig_shape[0] / img_small_shape[0]  # scale ration x
    ys = img_orig_shape[1] / img_small_shape[1]  # scale ration y
    x, y, w, h = box
    x = round((x + edge) * xs)
    y = round((y + edge) * ys)
    w = round(w * xs)
    h = round(h * ys)
    return (x,y,w,h)


def rotate(img_orig, angle, scale=1.0):
    center_orig = (img_orig.shape[1] // 2, img_orig.shape[0] // 2)
    rot_mat = cv.getRotationMatrix2D(center_orig, angle, scale)
    ret_img = cv.warpAffine(img_orig, rot_mat, (img_orig.shape[1], img_orig.shape[0]),
                            cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return ret_img

# def most_common_colours(img):
#     """
#     :param img:
#     :return: (223, 223, 223)
#     """
#     img = np.array(img)
#     smaller = cv.resize(img, (20, 20))
#     col = cv.split(smaller)
#     # print(len(col))
#     if len(col) == 1:  # gray
#         bv = (int(np.bincount(col[0][0]).argmax()))  # most common colours
#     else:
#         bv = (int(np.bincount(col[0][1]).argmax()), int(np.bincount(col[0][1]).argmax()),
#               int(np.bincount(col[0][2]).argmax()))
#     return bv


def get_lines_canny(img, k=1):
    """ HoughLines for 575x575 passports
    1) blur with k power
    2) find lines with treshhold 100, 70, 130
    3) return best of 100 or 70 or 130"""
    img2 = img.copy()
    img2 = img
    # img2 = cv.fastNlMeansDenoising(img2,  h=20, templateWindowSize=5)  # denoise edges
    # if k > 2:
    #     print(k // 2.5, k // 2.5)
    #     img2 = cv.blur(img2, (k // 2.5, k // 2.5))
    dilatation_type = cv.MORPH_RECT
    dilatation_size = 3
    element = cv.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    img2 = cv.erode(img2, element)
    img2 = cv.erode(img2, element)
    # img2 = cv.erode(img2, element)
    img2 = cv.dilate(img2, element)
    ret2, r = cv.threshold(img2, 0, 200, cv.THRESH_BINARY + cv.THRESH_OTSU)  # adaptive threshold
    edges = cv.Canny(r, 30 + 20 * math.sqrt(k / 2), 255, apertureSize=3)
    # edges = cv.Canny(r, 30 + 20 * (k / 2), 255)
    # edges = cv.Canny(r, 40, 100, apertureSize=3)

    lines = None

    h, w = edges.shape
    min_leng = w // 2.4  # 2.5 for 575
    gap = w // 0.01    # 4.4
    for tre in [100, 70, 130]:  # treshhold
        for i2 in range(7):  # less required line
            lines = cv.HoughLinesP(edges, 2, np.pi / 180, tre, minLineLength=min_leng, maxLineGap=gap / (0.6 * i2 + 0.3))
            # print(len(lines) if lines is not None else None, "i2", i2, "tre", tre)
            # if lines is not None:
            #     # print("LINES!!!!!!!", len(lines))
            #     img3 = img2.copy()
            #     for line in lines:
            #         x1, y1, x2, y2 = line[0]
            #         cv.line(img3, (x1, y1), (x2, y2), (0, 255, 0), 3)  # DEBUG!!
            #     #
            #     img2 = cv.resize(img3 , (900, 900))
            #     # plt.imshow(img)
            #     # plt.show()
            #     from matplotlib import pyplot as plt
            #     plt.imshow(img2)
            #     plt.savefig('/tmp/a.png')
            #     plt.close()

                # plt.show()
            #     # cv.imshow('image', img2)  # show image in window
            #     # cv.waitKey(0)  # wait for any key indefinitely
            #     # cv.destroyAllWindows()  # close window

            #     lines_count = len(lines)
            #     slopes = []
            #     for line in lines:
            #         line = line[0]
            #         x = line[0] - line[2]
            #         y = line[1] - line[3]
            #         slope = np.abs(x/y)
            #         print("slope", slope, line)
            #     # print(len(lines))
            if lines is not None:
                lines_count = len(lines)
                if 6 < len(lines) < 100:
                    # print("LINES!!!!!!!", len(lines))
                    break
        else:
            continue
        break
    return lines


def get_degree1(mr):
    """ abs(angle) < np.pi/2 """
    if mr < math.pi / 4:
        # print('30*')
        degree = - math.degrees(mr)  # I 30*
    else:
        # print('60 *')
        degree = math.degrees(math.pi / 2 - mr)  # I 60*
    return degree


def get_degree2(mr):
    """ abs(angle) >= np.pi/2 """
    if mr < 3 * math.pi / 4:
        # print('120 *')
        degree = - math.degrees(mr - math.pi / 2)  # III 120*
    else:
        # print('150 *')
        degree = math.degrees(math.pi - mr)  # III 150*
    return degree


def most_common(lst):
    """
    :param lst: iterable
    :return: one item
    """
    if lst:
        return max(set(lst), key=lst.count)
    else:
        return None


def get_lines_c(img):
    """ capthca recognition
    we define there li
    """
    LINE_LENGTH_MIN = 7
    LINE_LENGTH_NORM = 20
    if len(img.shape) > 2:
        img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        img2 = img.copy()
    lines = None
    before = None
    for i in range(15):  # less points
        # print(i)
        lines1 = get_lines_canny(img2, k=1 + i)  # loop to decrease lines count
        # print("lines1", len(lines1) if lines1 is not None else None)
        # print(lines1[0])
        # if lines1 is not None:
        #     print(len(lines1))
        # [[  9 107 488  14]]
        if lines1 is None or len(lines1) < LINE_LENGTH_MIN:  # too little
            if before is not None and len(before) < 40:  # OK use before
                lines = before
            break
        if lines1 is not None and len(lines1) < LINE_LENGTH_NORM:  # OK, stop
            lines = lines1
            break
        before = lines1

    # DEBUG!!

    # img = img.copy()
    # if lines is not None:
    #     print("LINES!!!!!!!", len(lines))
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # DEBUG!!
    #     #
    #     img2 = cv.resize(img, (400, 400))
    #     cv.imshow('image', img2)  # show image in window
    #     cv.waitKey(0)  # wait for any key indefinitely
    #     cv.destroyAllWindows()  # close window
            # from matplotlib import pyplot as plt
    #     plt.imshow(img)
    #     plt.show()
    return lines


# def get_lines_h(img): #deprecated
#     """ alternative rotate_image sum function
#     :param img: grayscale
#     :return:
#     """
#     if len(img.shape) > 2:
#         img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     h, w = img.shape
#     # print(w // 150)
#     th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, w // 150)
#     edges = cv.Canny(th, 70, 255, apertureSize=3)
#     leng = w // 2.5
#     lines = None
#     for i in range(20):
#         lines = cv.HoughLinesP(edges, 2, np.pi / 180, 100, minLineLength=leng - w // 40 * i, maxLineGap=25)
#         if lines is not None:
#             if len(lines) > 1:
#                 break
#
#     return lines


def rotate_detect(img, glf: Callable) -> (float):
    """ HoughLines magic

    :param img:
    :param gl: function must return lines cv.HoughLinesP
    :return: degree
    """
    # print("wtf1")
    lines = glf(img)

    angles1 = list()
    angles2 = list()

    degree = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(x2 - x1, y2 - y1)
            # print(x2 - x1)

            # if round(angle, 1) == 0.0:
            # cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # DEBUG!!
            if abs(angle) < np.pi / 2:  # radians
                angles1.append(angle)
            else:
                angles2.append(angle)

        # NEW
        degrees = [get_degree1(x) for x in angles1] + [get_degree2(x) for x in angles2]
        mc = most_common([round(a, 1) for a in degrees if abs(a) != 0])
        if mc is None:
            return 0
        filtered_degrees = []
        for a in degrees:
            if round(a, 1) == mc:
                filtered_degrees.append(a)

        med_degree = float(np.median(filtered_degrees))

        if med_degree is None:
            return 0
        else:
            return med_degree

    return degree


def find_angle(img_orig, glf: Callable) -> float:  # , copy=None
    """ Fix little angles
    1) image to 575 by width
    2) crop 30 pts by edges
    3) rotate image by degrees and find out angles with gl:Callable for every degree

    :param img_orig:
    :param gl:
    :return: image
    """
    img_small, _ = img_to_small(img_orig, height_target=575)
    ish = img_small.shape
    img_small = img_small[30:ish[0]-30, 30:ish[1]-30] # cut edges
    # print(img_small.shape)
    center_small = (img_small.shape[1] // 2, img_small.shape[0] // 2)

    def get_degree(angle): #, ret_list):
        # print("wtf")

        rot_mat = cv.getRotationMatrix2D(center_small, angle, scale=1)
        img_1 = cv.warpAffine(img_small, rot_mat, (img_small.shape[1], img_small.shape[0]),
                              borderMode=cv.BORDER_REFLECT)
        dr = rotate_detect(img_1, glf)
        ret = dr + angle
        # print("dr, angle, ret", dr, angle, ret)
        # ret_list.append(ret)
        if dr == 0:
            return None
        return ret

    degrees = [
        7,
        9,
        11,
        13,
        15,
        # 27,
        # 30
    ]
    degrees = degrees + [-x for x in degrees]

    import concurrent.futures
    angles: list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(get_degree, x): x for x in degrees}
        for future in concurrent.futures.as_completed(futures):
            # futures[future] # degree
            data = future.result()
            angles.append(data)


    bc = 0
    for d, a in zip(degrees, angles):
        if d == a:
            bc += 1
    er = bc / len(degrees)
    # print("errors rate:", er)
    if er == 1:
        return img_orig
    # print("angles", angles)
    a1 = most_common([round(a) for a in angles if a is not None])
    # print(a1)
    filtered_angles1 = []
    for a in angles :
        if a is not None and round(a) == a1:
            filtered_angles1.append(a)
    # print(filtered_angles1)
    return np.median(filtered_angles1)


def fix_angle(img_orig, gl: Callable = None, angle = None) -> np.array:  # , copy=None
    if gl is not None:
        angle = find_angle(img_orig, gl)
    if abs(angle) > 1:
        scale = 1.01
        return rotate(img_orig, angle, scale=scale)
    else:  # no blur of warpAffine
        return img_orig


def crop(img_input, rotate: bool = False, rate: float = 1) -> (np.ndarray, np.ndarray):
    """ Find object by contour area proportion(rate) to full image.
    Used Erosion before rectangle
     area not be reduced if nothing Dilation in the opposite direction.
    steps:
    1) denoising
    2) loop: findContours, find larges contour and get rectangle for it
    3) get subimage from random oriented rectangle
    4) (optional) rotate by fixing angles


    :param img_input:
    :param rotate:
    :param rate: 1 - passport 0.22 - driving license
    :return: BGR cropped and rotated, gray image without crop and not rotate
    """
    ratio_min = 0.40 * rate  # 0.357 # 0.42
    ratio_max = 0.571 * rate  # 1/1.75
    # resize to working size
    img_resized, scale_percent = img_to_small(img_input)
    img = img_resized

    # crop a little at the edge - very common noise
    try:
        h, w, _ = img.shape
    except:
        print("Image was readed wrong! Check Image path.")
        exit(1)
    edge = h // 38
    img = img[edge:(h - edge), edge:(w - edge)]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = gray
    # _, edges = cv.threshold(edges, 150, 255, cv.THRESH_TRUNC)
    # _, edges = cv.threshold(edges, 140, 255, cv.THRESH_BINARY)
    # img = cv.cornerHarris(img, 2, 3, 0.04)
    # edges = cv.Canny(img.copy(), 170, 255, apertureSize=5, L2gradient=False)  # edges

    img = cv.fastNlMeansDenoising(img, h=160, templateWindowSize=30)  # denoise edges
    img = cv.cornerHarris(img, 2, 3, 0.07)
    img = cv.dilate(img, None)

    ret, img = cv.threshold(img, 0.00001 * img.max(), 255, 0)
    # edges = cv.fastNlMeansDenoising(edges, h=120, templateWindowSize=10)  # denoise edges
    img = np.uint8(img)
    img = cv.fastNlMeansDenoising(img, h=80, templateWindowSize=20)  # denoise edges
    save_img = img
    total_area = img.shape[0] * img.shape[1]
    save_contour = None
    back_flag = False
    for i in range(150):
        # edges2 = edges
        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        area = 0
        for c in contours:
            area += cv.contourArea(c)
        kernel = np.ones((3, 3), np.uint8)

        if back_flag is False and (i > 140 or (area < total_area / 4.5 and i > 60)):  # 4.5 TUNNED PARAMETER!!!
            back_flag = True

        if back_flag:
            if save_contour is not None:
                break
            # _, edges2 = cv.threshold(edges, 0 + i * 2 - i, 255, 0)  # thresh
            img = cv.dilate(img, kernel, iterations=1)  # objects bigger

        else:
            _, img = cv.threshold(save_img, 0 + i * 2, 255, 0)  # thresh
            img = cv.dilate(img, kernel, iterations=15)  # objects bigger

        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = list(contours)

        for i in range(len(contours)):
            x, y, w, h = cv.boundingRect(contours[i])
            c = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])  # rectangle back to contour
            a = cv.contourArea(c)
            # print(a)
            if a < total_area * ratio_min:  # too small  # TUNNED PARAMETER!!!
                contours[i] = None
            if a > total_area * ratio_max:  # too large # 1.9  # TUNNED PARAMETER!!!
                contours[i] = None

        contours = np.array(list(filter(lambda x: x is not None, contours)))
        if len(contours) == 1:
            save_contour = contours[0]

    if save_contour is not None:
        # box = cv.boundingRect(save_contour)
        # x, y, w, h = scale_box(box, img_input.shape, img_resized.shape, edge)
        # img_input = img_input[y:(y + h), x:(x + w)]  # final crop

        def getSubImage(rect, src):
            # Get center, size, and angle from rect
            center, size, theta = rect
            # print(center, size, theta)
            if theta < -45:
                theta = theta + 90
                size = (size[1], size[0])

            # print(theta)
            if abs(theta) > 1:
                # Convert to int
                center, size = tuple(map(int, center)), tuple(map(int, size))
                # Get rotation matrix for rectangle
                M = cv.getRotationMatrix2D(center, theta, 1)
                # Perform rotation on src image - Causes the blurring!!!
                dst = cv.warpAffine(src, M, (src.shape[1], src.shape[0]), cv.INTER_CUBIC)
                out = cv.getRectSubPix(dst, size, center)
            else:
                width, height = (size[0], size[1])
                h1 = int(center[1] - height / 2)
                h2 = int(center[1] + height / 2)
                w1 = int(center[0] - width / 2)
                w2 = int(center[0] + width / 2)
                out = src[h1:h2, w1:w2]

                # debug
                # cv.line(src, (w1, h1), (0, 0), (0, 255, 0), 3)
                # img2 = cv.resize(src, (900, 900))
                # cv.imshow('image', img2)  # show image in window
                # cv.waitKey(0)  # wait for any key indefinitely
                # cv.destroyAllWindows()  # close window q
            return out

        rect = cv.minAreaRect(save_contour)

        rect = scale_rect(rect, img_input.shape, img_resized.shape, edge, scale_percent)

        # Debug
        # print(rect)
        # box_points = cv.boxPoints(rect)
        # print(box_points)
        #
        # box_points = np.intp(box_points)
        # # points = scale_points(points, img_input.shape, img_resized.shape, edge)
        # #
        # cv.drawContours(img_input, [box_points], 0, (0, 255, 0))
        #
        # import matplotlib.pyplot as plt
        # plt.imshow(img_input)
        # plt.show()
        # img2 = cv.resize(img_input, (900, 900))
        # cv.imshow('image', img2)  # show image in window
        # cv.waitKey(0)  # wait for any key indefinitely
        # cv.destroyAllWindows()  # close window q

        img_input = getSubImage(rect, img_input)

    if rotate:
        img_input = fix_angle(img_input, get_lines_c)  # get_lines_c or get_lines_h



    return img_input, gray

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

if __name__ == '__main__':
    # pass
    # p = '/mnt/hit4/hit4user/Desktop/passport_and_vod11.10_to_vod/45-176-1.png'
    # -------------- crop -------------
    # p = '/home/u/sources/tasks-for-job/task_for_zennolab/train_to_the_coordinates_dataset/2657.jpg'
    # img = cv.imread(p)
    # assert img is not None, "img could not be read"
    # img = img[0:200, 0:200]
    # img, _ = crop(img, rotate=True, rate=0.10)
    # img2 = cv.resize(img, (900, 900))
    # plt.imshow(img2)
    # plt.show()

    # ---------------- fix angle -----------
    from matplotlib import pyplot as plt

    p = '/home/u/sources/tasks-for-job/task_for_zennolab/train_to_the_coordinates_dataset/2749.jpg'
    img = cv.imread(p)
    assert img is not None, "img could not be read"
    img = img[0:200, 0:200]
    # img, _ = crop(img, rotate=False, rate=2.3)

    # r = get_lines_c(img)
    # img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # for i in range(25):  # less points
    #     # print(i)
    #     lines1 = get_lines_canny(img2, k=1 + i)
    #     print(lines1)
    # print(r)
    # img2 = cv.resize(img, (900, 900))
    # img = rotate(img, -6, 1.01)
    img = fix_angle(img, get_lines_c)
    plt.imshow(img)
    plt.show()


    # cv.imshow('Result', img2)
    # cv.waitKey()
    #img, degree = \
    # img = rotate(img, get_lines_c)
    # # print(degree)
    #
    #
    # # center = (img.shape[1] // 2, img.shape[0] // 2)
    # # scale = 1.03
    # # angle = -3
    # # rot_mat = cv.getRotationMatrix2D(center, angle, scale)
    # # img = cv.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), borderMode=cv.BORDER_REPLICATE)
    # #
    # # img, degree = rotate_image(img, get_lines_c)
    # # print(degree)
    #
    #
    # cv.imshow('Result', img)
    # cv.waitKey()
