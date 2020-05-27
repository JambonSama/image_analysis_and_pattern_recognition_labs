import gzip
import os
import numpy as np
from numpy.linalg import norm
import cv2 as cv
import pickle
from sklearn.neural_network import MLPClassifier 
import matplotlib.pyplot as plt
#%matplotlib inline
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
        detection

        """
        list = []
        for idx, l in enumerate(labels):
            if l == 9:
                list.append(idx)
        return np.delete(dataset, list, axis=0), np.delete(labels, list)

    def _extract_data(self, filename, image_shape, image_number):
        """
        Unzip the MNIST dataset and stores the data in memory
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
        Unzip the MNIST dataset and stores the labels in memory
        """
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1*image_number)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

    def _flatten_image(self, image):
        """
        Returns the image as 1 by 784 vector instead of 28*28
        """
        flattened_image = image.reshape(1, 
            self.image_shape[0]*self.image_shape[1])
        return flattened_image

    def _prepare_number_training_and_testing(self):
        """Extracts the datasets for use in the mlp
            stores the dataset without the character "9"
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
        Trains the MLP with the stored training data 
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
        Compute the score of the trained mlp against the test dataset
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
        Returns the predicted number from the mlp

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
        Returns the predicted sign from the mlp

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
    black = cv.inRange(hsv_frame, (0, 0, 0), (180, 255, 120))
    return red1+red2, blue, black


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

def detect_chars_pos_and_img(img, position_robot):
    _, blue_im, black_im = split_objects(img)
    im = blue_im + black_im
    cv.imshow("black",im)
    dilate_im = applyMorphology(im)
    contours, _ = cv.findContours(image=dilate_im, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    image_list = []
    position_list = []

    for contour in contours:
        if contour.shape[0] > 5:
            test = cv.fitEllipse(contour)
            (x, y), (MA, ma), angle = test
            if MA > 5 and ma > 15 and MA < 60 and ma < 60 and euclidian_distance((x, y), position_robot) > 50:
                position_list.append((x,y))
                rotation_matrix = cv.getRotationMatrix2D((x, y), angle, 1)
                rotated_im = cv.warpAffine(im, rotation_matrix, (img.shape[1], img.shape[0]))
                extract_im = rotated_im[int(y-ma/2)-4:int(y+ma/2)+4, int(x-MA/2)-4:int(x+MA/2)+4]
                
                rapport = int(np.round(28/extract_im.shape[0]*extract_im.shape[1]))
                resize_im = cv.resize(extract_im, (rapport, 28), interpolation=cv.INTER_LINEAR)
                left = int((28-resize_im.shape[1])/2)
                right = 28 - resize_im.shape[1] - left
                resize_im = cv.copyMakeBorder(resize_im, 0, 0, left, right, cv.BORDER_CONSTANT)
                print(resize_im.shape)
                _, threshold_im = cv.threshold(resize_im, 150, 255, cv.THRESH_BINARY)
                #threshold_im = applyMorphology2(threshold_im)
                image_list.append(threshold_im)
                cv.ellipse(img, test,(0,255,0),2)
            else:
                cv.ellipse(img, test,(255,0,0),2)

    cv.imshow("test", img)

    return image_list, position_list

def determine_chars(chars_img):
    """
    From the list of the characters images, returns a list of the characters,
    using the neural network for classification
    """
    ocr1 = OCR()
    # print(ocr1.compute_test_score())

    chars = []

    for image in chars_img:
        digit = str(ocr1.get_digit(image))
        sign = ocr1.get_sign(image)
        chars.append([digit, sign])

    return chars

def detect_chars(chars_img, robot_pos):
    """
    From a frame, returns a list of all the chars positions, and a list of all the chars.
    Frame is single channel image of the blue and black values (characters on the area).
    """
    #chars_pos, chars_img = detect_chars_pos_and_img(frame, robot_pos)
    chars = determine_chars(chars_img)
    return chars

def main():

    cv.destroyAllWindows()
    cap = cv.VideoCapture("../data/robot_parcours_1.avi")

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    if cap.isOpened():
        ret, frame = cap.read()
        frame = cv.imread("FirstImageMod.jpg")
        test = frame.copy()
        rotation_matrix = cv.getRotationMatrix2D((test.shape[1]/2, test.shape[0]/2), 55, 1)
        rotated_test = cv.warpAffine(test, rotation_matrix, (test.shape[1], test.shape[0]), borderValue=1)
        cv.imshow("Rotated image", rotated_test)
        
     
        images, positions = detect_chars_pos_and_img(frame, position_robot=(539, 354))
        chars = detect_chars(images, (539, 354))
   
        fig, axes = plt.subplots(1, len(images), figsize=(20, 1))
        for c, image in enumerate(images):
            axes[c].set_title(chars[c])
            axes[c].imshow(image, cmap='gray')
            axes[c].axis('off')
        plt.show()
      

    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
    print("Done")