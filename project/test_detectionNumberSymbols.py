import gzip
import os
import numpy as np
from numpy.linalg import norm
import cv2 as cv
import pickle
from sklearn.neural_network import MLPClassifier 
from skimage.measure import inertia_tensor
from skimage.measure import moments
import matplotlib.pyplot as plt
from main import*
#%matplotlib inline

def detect_chars_pos_and_img2(frame, robot_pos, base_image):
    """
    Detect where all the characters are in the frame and returns the list of it.
    Frame is single channel image of the blue and black values (characters on the area).
    """
    # Morphology to make the equal and divide one shape
    kernel = np.zeros((3, 3), np.uint8)
    cv.circle(img=kernel, center=(1, 1), radius=1, color=255, thickness=-1)
    dilated_img = cv.morphologyEx(frame, cv.MORPH_DILATE, kernel, iterations=4)

    # Find the contours of images
    contours, _ = cv.findContours(
        image=dilated_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    chars_pos = []
    chars_img = []

    for contour in contours:
        if contour.shape[0] > 5:
            # Find fit ellipse for each contour
            ellipse = cv.fitEllipse(contour)
            (x, y), (ma, MA), angle = ellipse

            # Verifie if the fit ellipse is big enough and not the robot
            r = range(5, 60)
            R = range(15, 60)
            robot_pos_threshold = 50
            if int(ma) in r and int(MA) in R and norm((x-robot_pos[0], y-robot_pos[1])) > robot_pos_threshold:

                # Rotate the image to allign the big axis verticaly
                rotation_matrix = cv.getRotationMatrix2D((x, y), angle, 1)
                rotated_img = cv.warpAffine(
                    frame, rotation_matrix, (frame.shape[1], frame.shape[0]))

                # Etract the char image for the rotated image
                extracted_img = rotated_img[int(
                    y-MA/2):int(y+MA/2), int(x-ma/2):int(x+ma/2)]
                
                # Resize the char image to have 28x28 pixel to by compatible with mlp
                rapport = int(np.round(20/extracted_img.shape[0]*extracted_img.shape[1]))
                print(extracted_img.shape)
                resize_img = cv.resize(extracted_img, (rapport, 20), interpolation=cv.INTER_LINEAR)
                left = int((28-resize_img.shape[1])/2)
                right = 28 - resize_img.shape[1] - left
                resize_img = cv.copyMakeBorder(resize_img, 4, 4, left, right, cv.BORDER_CONSTANT)

                # Threshold and morphology to fil holes and binarize with 0 or 255
                _, thresholded_img = cv.threshold(
                    resize_img, 150, 255, cv.THRESH_BINARY)
                # thresholded_img = cv.morphologyEx(
                #     thresholded_img, cv.MORPH_CLOSE, kernel, iterations=1)

                # Save the image and his position
                chars_img.append(thresholded_img)
                chars_pos.append((x, y))
                cv.ellipse(base_image, ellipse,(0,255,0),2)
            else:
                cv.ellipse(base_image, ellipse,(255,0,0),2)

    cv.imshow("test", base_image)

    return chars_pos, chars_img

def detect_chars_pos_and_img3(frame, robot_pos, base_image):
    """
    Detect where all the characters are in the frame and returns the list of it.
    Frame is single channel image of the blue and black values (characters on the area).
    """
    # Morphology to make the equal and divide one shape
    kernel = np.zeros((3, 3), np.uint8)
    cv.circle(img=kernel, center=(1, 1), radius=1, color=255, thickness=-1)
    dilated_img = cv.morphologyEx(frame, cv.MORPH_DILATE, kernel, iterations=4)

    # Find the contours of images
    contours, _ = cv.findContours(
        image=dilated_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    chars_pos = []
    chars_img = []


    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        r = range(10, 60)
        R = range(10, 60)
        robot_pos_threshold = 50
        if int(w) in r and int(h) in R and norm((x+w/2-robot_pos[0], y+h/2-robot_pos[1])) > robot_pos_threshold:
            pos_x = int(x+w/2)
            pos_y = int(y+h/2)
            chars_pos.append((pos_x, pos_y))
            extracted_img = frame[y:y+h, x:x+w]
            print(f"{extracted_img.shape}, {w}, {h}")
            if int(w) > int(h) :
                new_h = int(22/extracted_img.shape[1]*extracted_img.shape[0])
                resize_img = cv.resize(extracted_img, (22, new_h), interpolation=cv.INTER_LINEAR)
                top = int((28-resize_img.shape[0])/2)
                bottom = 28 - resize_img.shape[0] - top
                resize_img = cv.copyMakeBorder(resize_img, top, bottom, 3, 3, cv.BORDER_CONSTANT)
                
            else :
                new_w = int(22/extracted_img.shape[0]*extracted_img.shape[1])
                resize_img = cv.resize(extracted_img, (new_w, 22), interpolation=cv.INTER_LINEAR)
                left = int((28-resize_img.shape[1])/2)
                right = 28 - resize_img.shape[1] - left
                resize_img = cv.copyMakeBorder(resize_img, 3, 3, left, right, cv.BORDER_CONSTANT)
            
            mu = inertia_tensor(resize_img)
            moment = moments(resize_img)
            y_c = int(moment[1,0]/moment[0,0])
            x_c = int(moment[0,1]/moment[0,0])
            alpha = np.degrees(1/2 *np.arctan2(2*mu[0][1],(mu[1][1]-mu[0][0])))
            
            rotation_matrix = cv.getRotationMatrix2D((x_c, y_c), alpha, 1)
            rotated_img = cv.warpAffine(
                resize_img, rotation_matrix, (resize_img.shape[1],resize_img.shape[0]))
            translation_matrix = np.float32([[1,0,14-x_c],[0,1,14-y_c]])
            translated_img = cv.warpAffine(rotated_img, translation_matrix, (resize_img.shape[1],resize_img.shape[0]))

            _, thresholded_img = cv.threshold(
                    translated_img, 150, 255, cv.THRESH_BINARY)
            chars_img.append(thresholded_img)
            cv.circle(base_image, (pos_x, pos_y), radius=2, color=(0, 255, 0), thickness=-1)
    cv.imshow("Test bounding box", base_image)



    return chars_pos, chars_img


def main():

    cv.destroyAllWindows()
    cap = cv.VideoCapture("../data/robot_parcours_1.avi")
    r1 = [[10, 10], [170, 80], [170, 80]]  # first red (around 0+ hue)
    r2 = [[170, 10], [170, 80], [170, 80]]  # second red (around 0- hue)
    b1 = [[118, 20], [150, 60], [110, 60]]  # blue
    b2 = [[90, 90], [130, 130], [60, 60]]  # black

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    if cap.isOpened():
        ret, frame = cap.read()
        frame = cv.imread("FirstImageMod.jpg")
        
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        chars_img = split(hsv_frame, b1, b2)
        cv.imshow("black image", chars_img)
        
        positions, images = detect_chars_pos_and_img2(chars_img, (539, 354), frame)
        chars = determine_chars(images)
   
        fig, axes = plt.subplots(1, len(images), figsize=(20, 1))
        for c, image in enumerate(images):
            axes[c].set_title(chars[c][0]+", "+chars[c][1])
            axes[c].imshow(image, cmap='gray')
            axes[c].axis('off')
        plt.show()
      

    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
    print("Done")