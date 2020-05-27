#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
import cv2 as cv


def split(frame, b1, b2):
    """
    Splits values frome frame according to two sets of boundaries.
    Frame has 3 channels, and is in HSV space.
    """
    img1 = cv.inRange(frame, (b1[0][0]-b1[0][1], b1[1][0]-b1[1][1], b1[2][0]-b1[2][1]),
                      (b1[0][0]+b1[0][1], b1[1][0]+b1[1][1], b1[2][0]+b1[2][1]))
    img2 = cv.inRange(frame, (b2[0][0]-b2[0][1], b2[1][0]-b2[1][1], b2[2][0]-b2[2][1]),
                      (b2[0][0]+b2[0][1], b2[1][0]+b2[1][1], b2[2][0]+b2[2][1]))
    return img1+img2


def detect_arrow_position(frame):
    """
    Determines the position of the arrow in the frame.
    Frame is single channel image of the red values (arrow of robot).
    """
    # morphology to clean the img (a little)
    radius = 2
    size = 2*radius+1
    kernel = np.zeros((size, size), np.uint8)
    cv.circle(img=kernel, center=(radius, radius),
              radius=radius, color=255, thickness=-1)
    frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel)

    # connected components
    label_number, _, stats, centroids = cv.connectedComponentsWithStats(
        image=frame, connectivity=8)

    i = 1

    if label_number < 2:
        return (-1, -1), frame
    if label_number > 2:
        i = np.argmax(stats[1:, cv.CC_STAT_AREA])+1

    return (centroids[i][0], centroids[i][1])


def determine_chars(chars_img, neural_network):
    """
    From the list of the characters images, returns a list of the characters,
    using the neural network for classification
    """
    return 1


def detect_chars_pos_and_img(frame, robot_pos):
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
            (x, y), (MA, ma), angle = cv.fitEllipse(contour)

            # Verifie if the fit ellipse is big enough and not the robot
            r = range(15, 60)
            if int(MA) in r and int(ma) in r and norm((x-robot_pos[0], y-robot_pos[1])) > 20:

                # Rotate the image to allign the big axis verticaly
                rotation_matrix = cv.getRotationMatrix2D((x, y), angle, 1)
                rotated_im = cv.warpAffine(
                    frame, rotation_matrix, (frame.shape[1], frame.shape[0]))

                # Etract the char image for the rotated image
                extracted_img = rotated_im[int(
                    y-ma/2)-4:int(y+ma/2)+4, int(x-MA/2)-4:int(x+MA/2)+4]

                # Resize the char image to have 28x28 pixel to by compatible with mlp
                resized_img = cv.resize(
                    extracted_img, (28, 28), interpolation=cv.INTER_AREA)

                # Threshold and morphology to fil holes and binarize with 0 or 255 
                _, thresholded_img = cv.threshold(
                    resized_img, 150, 255, cv.THRESH_BINARY)
                thresholded_img = cv.morphologyEx(
                    thresholded_img, cv.MORPH_CLOSE, kernel, iterations=1)
                
                # Save the image and his position
                chars_img.append(thresholded_img)
                chars_pos.append((x, y))

    return chars_pos, chars_img


def detect_chars(frame, robot_pos, neural_network):
    """
    From a frame, returns a list of all the chars positions, and a list of all the chars.
    Frame is single channel image of the blue and black values (characters on the area).
    """
    chars_pos, chars_img = detect_chars_pos_and_img(frame, robot_pos)
    chars = determine_chars(chars_img, neural_network)

    chars = np.arange(len(chars_pos))
    return chars_pos, chars


def main():
    """
    Main function of program.
    """
    cap = cv.VideoCapture("../data/robot_parcours_1.avi")

    # HSV boundaries for color detection
    r1 = [[10, 10], [170, 80], [170, 80]]  # first red (around 0+ hue)
    r2 = [[170, 10], [170, 80], [170, 80]]  # second red (around 0- hue)
    b1 = [[118, 20], [150, 60], [110, 60]]  # blue
    b2 = [[90, 90], [125, 125], [60, 60]]  # black

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Frame counter
    i = 0

    # Lists of symbols
    formula = []
    robot_pos = []

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Dimensions of the frame
            height = frame.shape[0]

            # Conversion to HSV space for better color separation
            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            # Find the arrow position
            arrow_img = split(hsv_frame, r1, r2)
            robot_pos.append(detect_arrow_position(arrow_img))

            if i == 0:
                chars_img = split(hsv_frame, b1, b2)
                chars_pos, chars = detect_chars(
                    chars_img, robot_pos[0], neural_network=1)

            # Retrieve the formula
            for i, pos in enumerate(chars_pos):
                # if robot passes over char
                if norm((pos[0]-robot_pos[-1][0], pos[1]-robot_pos[-1][1])) < 35:
                    formula.append(chars[i])  # add char to formula
                    # delete char from list so as to not add twice while in the same circle
                    del chars_pos[i]
                    chars = np.delete(chars, i)  # same as above

                # Highlight all the chars
                cv.circle(frame, (int(pos[0]), int(pos[1])), radius=2,
                          color=(0, 255, 0), thickness=-1)

            # Highlight the robot positions
            cv.circle(frame, center=(int(robot_pos[-1][0]), int(robot_pos[-1][1])),
                      radius=5, color=(0, 0, 255), thickness=-1)
            for i, _ in enumerate(robot_pos):
                if i > 0:
                    p1 = (int(robot_pos[i-1][0]), int(robot_pos[i-1][1]))
                    p2 = (int(robot_pos[i][0]), int(robot_pos[i][1]))
                    cv.line(frame, p1, p2, color=(0, 0, 255), thickness=3)

            # Write info
            info = f"Formula : {formula}"
            cv.putText(img=frame, text=info, org=(10, height-10), fontFace=cv.FONT_HERSHEY_PLAIN,
                       fontScale=1.5, color=(255, 0, 0), thickness=2)

            # Display the "video"
            cv.imshow("IAPR project", frame)

            # Press Q on keyboard to  exit
            if cv.waitKey(300) == ord("q"):
                break

            # Frame counter
            i += 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Print the result
    print(f"{formula}")

    # Closes all the frames
    cv.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
    print("Done")
