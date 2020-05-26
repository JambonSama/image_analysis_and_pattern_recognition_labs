import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
#%matplotlib inline


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
    black = cv.inRange(hsv_frame, (0, 0, 0), (180, 250, 120))
    return red1+red2, blue, black

def detect_arrow_position(frame):
    """
    Determines the position of the arrow in the frame.
    """
    pos = (5, 4)
    return pos, frame

def applyMorphology(img):
    kernel = np.zeros((3,3), np.uint8)
    cv.circle(img=kernel, center=(1,1), radius=1, color=255, thickness=-1)
    result = img.copy()
    result = cv.morphologyEx(img, cv.MORPH_DILATE, kernel, iterations=4)
    return result

def applyMorphology2(img):
    kernel = np.zeros((3,3), np.uint8)
    cv.circle(img=kernel, center=(1,1), radius=1, color=255, thickness=-1)
    result = img.copy()
    result = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=1)
    return result

def euclidian_distance(pos1,pos2):
    return (np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2))

def functionTransform(img):
    x = []
    y = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != 0 :
                x.append(i)
                y.append(j)
               
    xa = np.array(x)
    ya = np.array(y)
      
    return xa, ya


def detectNumbersSymbols(img, nom, position_robot):
    _, blue_im, black_im = split_objects(img)
    im = blue_im + black_im
    dilate_im = applyMorphology(im)
    contours, _ = cv.findContours(image=dilate_im, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    image_list = []
    position_list = []

    for contour in contours:
        rect = cv.minAreaRect(contour)    
        (x, y), (width, height), angle = rect    
        if width > 15 and height > 15 and width < 40 and height < 40 and euclidian_distance((x, y), position_robot) > 20:
            position_list.append((x, y))
            rotation_matrix = cv.getRotationMatrix2D((x, y), angle, 1)
            rotated_im = cv.warpAffine(im, rotation_matrix, (img.shape[1], img.shape[0]))
            extract_im = rotated_im[int(y-height/2):int(y+height/2), int(x-width/2):int(x+width/2)]
            
            resize_im = extract_im.copy()
           
            if extract_im.shape[0] < 28 :
                top = int((28-extract_im.shape[0])/2)
                bottom = 28-extract_im.shape[0]-top
                resize_im = cv.copyMakeBorder(resize_im, top, bottom, 0, 0, cv.BORDER_CONSTANT)
            if extract_im.shape[1] < 28 :
                left = int((28-extract_im.shape[1])/2)
                right = 28-extract_im.shape[1]-left
                resize_im = cv.copyMakeBorder(resize_im, 0, 0, left, right, cv.BORDER_CONSTANT)
            if extract_im.shape[0] > 28 or extract_im.shape[1] > 28 :
                resize_im = cv.resize(extract_im, (28, 28), interpolation=cv.INTER_AREA)
           

            _, threshold_im = cv.threshold(resize_im,150,255,cv.THRESH_BINARY)
            threshold_im = applyMorphology2(threshold_im)
            image_list.append(threshold_im*255)

            # Print box detection
            box = cv.boxPoints(rect)
            cv.drawContours(img, [np.intp(box)], 0, [0, 255, 0])
            cv.circle(img, (int(x),int(y)), radius=2, color=(0,0,255), thickness=-1)
        else :
            box = cv.boxPoints(rect)
            cv.drawContours(img, [np.intp(box)], 0, [255, 0, 0])
    cv.imshow(nom, img)
    return image_list, position_list

def detectNumbersSymbols2(img):
    _, blue_im, black_im = split_objects(img)
    im = blue_im + black_im
    dilate_im = applyMorphology(im)
    contours, _ = cv.findContours(image=dilate_im, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    image_list = []
    position_list = []

    for contour in contours:
        (x, y), (MA,ma),angle = cv.fitEllipse(contour)
        
    cv.imshow("test", img)

    return image_list, position_list

def main():

    cv.destroyAllWindows()
    cap = cv.VideoCapture("../data/robot_parcours_1.avi")

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    if cap.isOpened():
        ret, frame = cap.read()

        test = frame.copy()
        rotation_matrix = cv.getRotationMatrix2D((test.shape[1]/2, test.shape[0]/2), 45, 1)
        rotated_test = cv.warpAffine(test, rotation_matrix, (test.shape[1], test.shape[0]))
        cv.imshow("Rotated image", rotated_test)
        
        images, positions = detectNumbersSymbols(frame, 'normal', (0,0))
    
        images_r, positions_r = detectNumbersSymbols(rotated_test, 'rotated', (0,0))
        print(positions_r[6])
        fig, axes = plt.subplots(2, len(images), figsize=(12, 3))
        for c, image in enumerate(images):
            axes[0][c].imshow(image, cmap='gray')
            axes[0][c].axis('off')
      
        for c, image in enumerate(images_r):
            axes[1][c].imshow(image, cmap='gray')
            axes[1][c].axis('off')
        plt.show()
    # When everything done, release the video capture object
    while cv.waitKey(300) != ord("q"):
        time.sleep(1)
    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
    print("Done")