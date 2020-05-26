
def split_objects(frame):
    """
    Splits the image between the three types of objects of interest.
    """
    R_H1 = 10
    R_H2 = 170
    R_S = 170
    R_V = 170

    B_H = 118
    B_S = 150
    B_V = 110
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    red1 = cv.inRange(hsv_frame, (R_H1-10, R_S-80, R_V-80), (R_H1+10, R_S+80, R_V+80))
    red2 = cv.inRange(hsv_frame, (R_H2-10, R_S-80, R_V-80), (R_H2+10, R_S+80, R_V+80))
    blue = cv.inRange(hsv_frame, (B_H -20, B_S-60, B_V-60), (B_H +20, B_S+60, B_V+60))
    black = cv.inRange(hsv_frame, (0, 0, 0), (180, 250, 100))
    return red1+red2, blue, black

def applyMorphology(img):
    kernel = np.zeros((3,3), np.uint8)
    cv.circle(img=kernel, center=(1,1), radius=1, color=255, thickness=-1)
    result = img.copy()
    result = cv.morphologyEx(img, cv.MORPH_DILATE, kernel, iterations=5)
    return result

def detectNumbersSymbols(img):
    _, blue_im, black_im = split_objects(img)
    im = blue_im + black_im
    dilate_im = applyMorphology(im)
    contours, _ = cv.findContours(image=dilate_im, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    image_list = []
    position_list = []

    for contour in contours:
        rect = cv.minAreaRect(contour)    
        (x, y), (width, height), angle = rect    
        if width > 15 and height > 15 and width < 100 and height < 100:
            position_list.append((x, y))
            rotation_matrix = cv.getRotationMatrix2D((x, y), angle, 1)
            rotated_im = cv.warpAffine(im, rotation_matrix, (img.shape[1], img.shape[0]))
            extract_im = rotated_im[int(y-height/2):int(y+height/2), int(x-width/2):int(x+width/2)]
            resize_im = cv.resize(extract_im, (28, 28), interpolation=cv.INTER_AREA)
            image_list.append(resize_im)

    #         # Print box detection
    #         box = cv.boxPoints(rect)
    #         cv.drawContours(img, [np.intp(box)], 0, [0, 255, 0])
    #         cv.circle(img, (int(x),int(y)), radius=2, color=(0,0,255), thickness=-1)
    #     else :
    #         box = cv.boxPoints(rect)
    #         cv.drawContours(img, [np.intp(box)], 0, [255, 0, 0])
    # cv.imshow("Box detection", img)
    return image_list, position_list