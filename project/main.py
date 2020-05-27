#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import os
import pickle
import argparse
import numpy as np
from numpy.linalg import norm
import cv2 as cv
from sklearn.neural_network import MLPClassifier

class OCR:
    """
    Converts an image into a number or a sign
    Doesn't detect if it is a sign or a number.

    """
    def __init__(self):
        self.model = []


        self.image_shape = (28,28)

        self.data_base_path = os.path.join(os.pardir, 'data')
        self.data_folder = 'lab-03-data'

        self._prepare_number_training_and_testing()
        self._train_classifiers()

    def _remove_nine(self, dataset, labels):
        """
        Remove the nine "9" character from the given dataset
        As the nine is not used for this lab, we remove it to avoid false 
        detection.

        """
        idx_list = []
        for idx, l in enumerate(labels):
            if l == 9:
                idx_list.append(idx)
        return np.delete(dataset, idx_list, axis=0), np.delete(labels, idx_list)

    def _extract_data(self, filename, image_shape, image_number):
        """
        Unzip the MNIST dataset and stores the data in memory.
        """
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(np.prod(image_shape)*image_number)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(image_number, image_shape[0], 
                image_shape[1])
        return data

    def _extract_labels(self, filename, image_number):
        """
        Unzip the MNIST dataset and stores the labels in memory.
        """
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1*image_number)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

    def _flatten_image(self, image):
        """
        Returns the image as 1 by 784 vector instead of 28*28.
        """
        flattened_image = image.reshape(1, 
            self.image_shape[0]*self.image_shape[1])
        return flattened_image

    def _prepare_number_training_and_testing(self):
        """
        Extracts the datasets for use in the mlp
        stores the dataset without the character "9".
        """
        train_set_size = 60000
        test_set_size = 10000

        data_part2_folder = os.path.join(self.data_base_path, 
            self.data_folder, 'part2')
        train_images_path = os.path.join(data_part2_folder, 
            'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_part2_folder, 
            'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_part2_folder, 
            't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_part2_folder, 
            't10k-labels-idx1-ubyte.gz')

        train_images = self._extract_data(train_images_path, self.image_shape, 
            train_set_size)
        test_images = self._extract_data(test_images_path, self.image_shape,
            test_set_size)
        self.train_labels = self._extract_labels(train_labels_path, 
            train_set_size)
        self.test_labels = self._extract_labels(test_labels_path, 
            test_set_size)
        self.flattened_training_set = train_images.reshape(
            train_set_size, self.image_shape[0]*self.image_shape[1])
        self.flattened_testing_set = test_images.reshape(test_set_size,
            self.image_shape[0]*self.image_shape[1])

        self.flattened_training_set, self.train_labels = self._remove_nine(
            self.flattened_training_set, self.train_labels)

    def _train_classifiers(self):
        """
        Trains the MLP with the stored training data.
        """
        # Definition of the used models
        if os.path.isfile('model.mlp'):
            print("Already trained model found")
            self.model = pickle.load(open('model.mlp','rb'))
        else:
            print("No previous model found.\nTraining new model and",
                " saving it to ./model.mlp")

            self._prepare_number_training_and_testing()
            self.model = MLPClassifier(hidden_layer_sizes=(100, ), 
                activation='relu', solver='adam', alpha=1e-3, 
                batch_size = 'auto', learning_rate = 'constant', 
                max_iter=2000, shuffle = True, random_state=1, tol=0.00001, 
                verbose = False, warm_start = False, early_stopping=False, 
                validation_fraction=0.1, beta_1=0.9, beta_2=0.99999, 
                epsilon=1e-08, n_iter_no_change=10)

            self.model.fit(self.flattened_training_set, self.train_labels)

            pickle.dump(self.model, open('model.mlp','wb'))
        
    def compute_test_score(self):
        """
        Compute the score of the trained mlp against the test dataset.
        """
        output = self.model.predict(self.flattened_testing_set)
        nb_wrong = 0
        idx_list = []
        for idx, predicted in enumerate(output):
            if predicted != self.test_labels[idx]:
                nb_wrong = nb_wrong + 1 
                idx_list.append(idx)

        return 1-(nb_wrong/10000)

    def get_digit(self, image):
        """
        Returns the predicted number from the mlp.

        image: 28 by 28 binary (1 and 0) matrix
        return: char of the detected value
        """
        angle = 180
        rotation_matrix = cv.getRotationMatrix2D(
            (self.image_shape[0]/2, self.image_shape[1]/2), angle, 1)
        rotated_im = cv.warpAffine(image, rotation_matrix, 
            (self.image_shape[1], self.image_shape[0]))

        im_fl = self._flatten_image(image).astype('float32')
        im_fl_reverse = self._flatten_image(rotated_im).astype('float32')


        prediction_normal = self.model.predict(im_fl)
        probability_normal = self.model.predict_proba(im_fl)

        prediction_reverse = self.model.predict(im_fl_reverse)
        probability_reverse = self.model.predict_proba(im_fl_reverse)

        if probability_reverse[0,prediction_reverse] \
            > probability_normal[0,prediction_normal]:
            return prediction_reverse[0]
        else: 
            return prediction_normal[0]

    def get_sign(self, image):
        """
        Returns the predicted sign from the mlp.

        image: 28 by 28 binary (1 and 0) matrix
        return: char of the detected sign
        """

        # Start by counting the number of contours
        # if 3 => division, 2 => equal, 1 => other
        num_shapes,_ = cv.connectedComponents(image=image)

        if (num_shapes - 1) == 3:
            return "/"
        elif (num_shapes - 1) == 2:
            return "="
        else:
            if np.sum(image == 1) > 170:
                return "*"
            else:
                return "+"


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


def determine_chars(chars_img):
    """
    From the list of the characters images, returns a list of the characters,
    using the MLP for classification.
    """
    ocr1 = OCR()
    # print(ocr1.compute_test_score())

    chars = []

    for image in chars_img:
        digit = str(ocr1.get_digit(image))
        sign = ocr1.get_sign(image)
        chars.append([digit, sign])

    return chars


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
            r = range(15, 60)
            R = range(5, 60)
            robot_pos_threshold = 35
            if int(ma) in r and int(MA) in R and norm((x-robot_pos[0], y-robot_pos[1])) > robot_pos_threshold:

                # Rotate the image to allign the big axis verticaly
                rotation_matrix = cv.getRotationMatrix2D((x, y), angle, 1)
                rotated_img = cv.warpAffine(
                    frame, rotation_matrix, (frame.shape[1], frame.shape[0]))

                # Etract the char image for the rotated image
                extracted_img = rotated_img[int(
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


def detect_chars(frame, robot_pos):
    """
    From a frame, returns a list of all the chars positions, and a list of all the chars.
    Frame is single channel image of the blue and black values (characters on the area).
    """
    chars_pos, chars_img = detect_chars_pos_and_img(frame, robot_pos)
    chars = determine_chars(chars_img)
    return chars_pos, chars


def main(input_filename, output_filename):
    """
    Main function of program.
    """
    capture_video = cv.VideoCapture(input_filename)

    # HSV boundaries for color detection
    r1 = [[10, 10], [170, 80], [170, 80]]  # first red (around 0+ hue)
    r2 = [[170, 10], [170, 80], [170, 80]]  # second red (around 0- hue)
    b1 = [[118, 20], [150, 60], [110, 60]]  # blue
    b2 = [[90, 90], [125, 125], [60, 60]]  # black

    # Check if camera opened successfully
    if not capture_video.isOpened():
        print("Error opening video stream or file.")
        return

    # Configuring output video encoding
    width, height = int(capture_video.get(3)), int(capture_video.get(4))
    fourcc = cv.VideoWriter_fourcc('F', 'M', 'P', '4')
    output_video = cv.VideoWriter(output_filename, fourcc, 2.0, (width, height))

    # Frame counter
    i = 0

    # Lists of symbols
    formula = []
    robot_pos = []

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
                chars_pos, chars = detect_chars(chars_img, robot_pos[0])

            # Retrieve the formula
            for i, pos in enumerate(chars_pos):
                # if robot passes over char
                robot_pos_threshold = 35
                if norm((pos[0]-robot_pos[-1][0], pos[1]-robot_pos[-1][1])) < robot_pos_threshold:
                    if len(formula) % 2 == 0:
                        formula.append(chars[i][0])  # add char to formula
                    else:
                        formula.append(chars[i][1])
                    # delete char from list so as to not add twice while in the same circle
                    del chars_pos[i]
                    del chars[i] # same as above

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
    print(f"{formula}")

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
