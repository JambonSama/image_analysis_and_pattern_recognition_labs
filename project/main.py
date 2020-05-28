#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import argparse
import numpy as np
from numpy.linalg import norm
import cv2 as cv
import optical_character_recognizer as ocr

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


def determine_chars(chars_img, cnn):
    """
    From the list of the characters images, returns a list of the characters,
    using the CNN for classification.
    """
    chars = []

    for image in chars_img:
        digit = str(cnn.get_digit(image))
        sign = cnn.get_sign(image)
        chars.append([digit, sign])

    return chars

def normalize_img(img):
    rect = cv.boundingRect(img)
    length = np.max(rect[2:])
    ratio = 20/length
    img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    img = cv.resize(src=img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv.INTER_LINEAR)
    height, width = img.shape[:2]
    moments = cv.moments(img)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    oX = 14-cX
    oY = 14-cY
    final_img = np.zeros((28, 28), np.uint8)
    final_img[oY:oY+height, oX:oX+width] = img
    return final_img

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
            (x, y), (ma, MA), angle = cv.fitEllipse(contour)

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
                    y-MA/2):int(y+MA/2), int(x-ma/2)-4:int(x+ma/2)]

                # Normalize image like MNIST
                normalized_img = normalize_img(extracted_img)

                # Save the image and his position
                chars_img.append(normalized_img)
                chars_pos.append((x, y))

    return chars_pos, chars_img


def detect_chars(frame, robot_pos, cnn):
    """
    From a frame, returns a list of all the chars positions, and a list of all the chars.
    Frame is single channel image of the blue and black values (characters on the area).
    """
    chars_pos, chars_img = detect_chars_pos_and_img(frame, robot_pos)
    chars = determine_chars(chars_img, cnn)
    return chars_pos, chars

def main(input_filename, output_filename):
    """
    Main function of program.
    """
    capture_video = cv.VideoCapture(input_filename)

    cnn = ocr.DigitNet()
    cnn.load_state_dict(torch.load("./mnist_net.cnn"))


    # HSV boundaries for color detection
    r1 = [[10, 10], [170, 80], [170, 80]]  # first red (around 0+ hue)
    r2 = [[170, 10], [170, 80], [170, 80]]  # second red (around 0- hue)
    b1 = [[118, 20], [150, 60], [110, 60]]  # blue
    b2 = [[90, 90], [130, 130], [60, 60]]  # black

    # Check if camera opened successfully
    if not capture_video.isOpened():
        print("Error opening video stream or file.")
        return

    # Configuring output video encoding
    width, height = int(capture_video.get(3)), int(capture_video.get(4))
    fourcc = cv.VideoWriter_fourcc('F', 'M', 'P', '4')
    output_video = cv.VideoWriter(
        output_filename, fourcc, 2.0, (width, height))

    # Frame counter
    i = 0

    # Lists of symbols
    robot_pos = []
    formula = ""

    # Read until video is completed
    while capture_video.isOpened():
        # Capture frame-by-frame
        ret, frame = capture_video.read()
        if ret:
            # Conversion to HSV space for better color separation
            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            # Find the arrow position
            arrow_img = split(hsv_frame, r1, r2)
            robot_pos.append(detect_arrow_position(arrow_img))

            if i == 0:
                chars_img = split(hsv_frame, b1, b2)
                chars_pos, chars = detect_chars(chars_img, robot_pos[0], cnn)

            # Retrieve the formula
            for i, pos in enumerate(chars_pos):
                # if robot passes over char
                robot_pos_threshold = 35
                if norm((pos[0]-robot_pos[-1][0], pos[1]-robot_pos[-1][1])) < robot_pos_threshold:
                    if len(formula) % 2 == 0:
                        formula += chars[i][0]  # add char to formula
                    else:
                        formula += chars[i][1]
                        if formula[-1] == '=':
                            formula += f"{eval(formula[:-1]):0.3f}"
                    # delete char from list so as to not add twice while in the same circle
                    del chars_pos[i]
                    del chars[i]  # same as above

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
                    cv.line(frame, p1, p2, color=(0, 0, 255), thickness=2)

            # Write info
            info = f"Formula : {formula}"
            cv.putText(img=frame, text=info, org=(10, height-10), fontFace=cv.FONT_HERSHEY_PLAIN,
                       fontScale=1.5, color=(255, 0, 0), thickness=2)

            # Display the "video"
            cv.imshow("IAPR project", frame)
            output_video.write(frame)

            # Press Q on keyboard to  exit
            if cv.waitKey(300) == ord("q"):
                break

            # Frame counter
            i += 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    capture_video.release()
    output_video.release()

    # Print the result
    print(f"Formula : {formula}")

    # Closes all the frames
    cv.destroyAllWindows()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the video file.")
    parser.add_argument("input", nargs=1, default="../data/robot_parcours_1.avi", type=str,
                        help="Input file path.", metavar="<input file filename>")
    parser.add_argument("-o", "--output", nargs=1, default="output.avi", type=str,
                        help="Output file path.", metavar="<output file filename>")
    args = parser.parse_args()

    main(args.input[0], args.output[0])
    print("Done")
