#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv

R_H1 = 10
R_H2 = 170
R_S = 170
R_V = 170

B_H = 118
B_S = 150
B_V = 110

def split_objects(frame):
    """
    Splits the image between the three types of objects of interest.
    """
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    red1 = cv.inRange(hsv_frame, (R_H1-10, R_S-80, R_V-80), (R_H1+10, R_S+80, R_V+80))
    red2 = cv.inRange(hsv_frame, (R_H2-10, R_S-80, R_V-80), (R_H2+10, R_S+80, R_V+80))
    blue = cv.inRange(hsv_frame, (B_H -20, B_S-60, B_V-60), (B_H +20, B_S+60, B_V+60))
    black = cv.inRange(hsv_frame, (0, 0, 0), (180, 250, 50))
    return red1+red2, blue, black

def detect_arrow_position(frame):
    """
    Determines the position of the arrow in the frame.
    """
    pos = (5, 4)
    return pos, frame

def main():
    """
    Main function of program.
    """
    cap = cv.VideoCapture("../data/robot_parcours_1.avi")

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Separation of robot arrow, symbols, and digits
            arrow, symbols, digits = split_objects(frame)

            _, arrow = detect_arrow_position(arrow)

            # Display the resulting frames
            cv.imshow("red", arrow)
            cv.imshow("blue", symbols)
            cv.imshow("black", digits)

            # Press Q on keyboard to  exit
            if cv.waitKey(300) == ord("q"):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()
    return

if __name__ == "__main__":
    main()
    print("Done")
