#!/usr/bin/env python
# ne pas enlever la ligne 1, ne pas mettre de commentaire au dessus
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import cv2 as cv
import time
import pickle
import matplotlib.pyplot as plt
#%matplotlib inline

# image definitions
image_shape = (28, 28)

class OCR:

    def __init__(self):
        self.model = []

        self.data_base_path = os.path.join(os.pardir, 'data')
        self.data_folder = 'lab-03-data'
        self.data_part2_folder = os.path.join(self.data_base_path, self.data_folder, 'part2')
        self.train_images_path = os.path.join(self.data_part2_folder, 'train-images-idx3-ubyte.gz')
        self.train_labels_path = os.path.join(self.data_part2_folder, 'train-labels-idx1-ubyte.gz')
        self.test_images_path = os.path.join(self.data_part2_folder, 't10k-images-idx3-ubyte.gz')
        self.test_labels_path = os.path.join(self.data_part2_folder, 't10k-labels-idx1-ubyte.gz')

        self.prepare_number_training_and_testing()
        self.trainClassifiers()

    def remove_nine(self, dataset, labels):
    	list = []
    	for idx, l in enumerate(labels):
    		if l == 9:
    			list.append(idx)
    	return np.delete(dataset, list, axis=0), np.delete(labels, list)

    def extract_data(self, filename, image_shape, image_number):
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(np.prod(image_shape) * image_number)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(image_number, image_shape[0], image_shape[1])
        return data

    def extract_labels(self, filename, image_number):
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * image_number)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

    def flatten_image(self, image):
        flattened_image = image.reshape(1, image_shape[0]*image_shape[1])
        return flattened_image

    def prepare_number_training_and_testing(self):
        train_set_size = 60000
        test_set_size = 10000

        train_images = self.extract_data(self.train_images_path, image_shape, train_set_size)
        test_images = self.extract_data(self.test_images_path, image_shape, test_set_size)
        self.train_labels = self.extract_labels(self.train_labels_path, train_set_size)
        self.test_labels = self.extract_labels(self.test_labels_path, test_set_size)
        self.flattened_training_set = train_images.reshape(train_set_size, image_shape[0]*image_shape[1])
        self.flattened_testing_set = test_images.reshape(test_set_size,image_shape[0]*image_shape[1])

        self.flattened_training_set, self.train_labels = self.remove_nine(self.flattened_training_set, self.train_labels)

    def trainClassifiers(self):
        #definition of the used models
        if os.path.isfile('model.mlp'):
            print("Already trained model found")
            self.model = pickle.load(open('model.mlp','rb'))
        else:
            print("No previous model found.\nTraining new model and saving it to ./model.mlp")

            self.prepare_number_training_and_testing()
            self.model = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=1e-3, batch_size='auto', 
                            learning_rate='constant', max_iter=2000, shuffle=True, random_state=1, 
                            tol=0.00001, verbose=False, warm_start=False, early_stopping=False, 
                            validation_fraction=0.1, beta_1=0.9, beta_2=0.99999, epsilon=1e-08, n_iter_no_change=10)
            
            self.model.fit(self.flattened_training_set, self.train_labels)

            pickle.dump(self.model, open('model.mlp','wb'))
        
    def computeTestScore(self):
        output = self.model.predict(self.flattened_testing_set)
        nb_wrong = 0
        idx_list = []
        for idx, predicted in enumerate(output):
            if predicted != self.test_labels[idx]:
                nb_wrong = nb_wrong +1 
                idx_list.append(idx)

        return 1-(nb_wrong/10000)

    def get_number(self, image):
        #predictions_proba = self.model.predict_proba(self.flatten_image(image).astype('float32')*255)
        predictions = self.model.predict(self.flatten_image(image).astype('float32')*255)

        return predictions[0]

    def get_sign(self, image):
        # start by counting the number of contours
        # if 3 => division, 2 => equal, 1 => other
        num_shapes,_ = cv.connectedComponents(image=image)
        #print(num_shapes-1)
        if num_shapes-1 == 3:
            return "/"
        elif num_shapes-1 == 2:
            return "="
        else:
            if np.sum(image == 1) > 170:
                return "*"
            else:
                return "+"


def get_eqn_result(string):
    # remove the equal sign before the eval. 
    return eval(string.replace("=","")) 

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

def euclidian_distance(pos1,pos2):
    return (np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2))

def applyMorphology2(img):
    kernel = np.zeros((3,3), np.uint8)
    cv.circle(img=kernel, center=(1,1), radius=1, color=255, thickness=-1)
    result = img.copy()
    result = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=1)
    return result

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
        if width > 15 and height > 15 and width < 40 and height < 40 and euclidian_distance((x, y), position_robot)> 20:
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
        
def main():
    ocr1 = OCR()
    print(ocr1.computeTestScore())

    #cv.destroyAllWindows()
    cap = cv.VideoCapture("../data/robot_parcours_1.avi")

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    if cap.isOpened():
        ret, frame = cap.read()

        test = frame.copy()
        rotation_matrix = cv.getRotationMatrix2D((test.shape[1]/2, test.shape[0]/2), 45, 1)
        rotated_test = cv.warpAffine(test, rotation_matrix, (test.shape[1], test.shape[0]))
        #cv.imshow("Rotated image", rotated_test)
        
        images, positions = detectNumbersSymbols(frame, 'normal',(0,0))

        images_r,_ = detectNumbersSymbols(rotated_test, 'rotated',(0,0))
        fig, axes = plt.subplots(2, len(images), figsize=(12, 3))
        for c, image in enumerate(images):
            axes[0][c].imshow(image, cmap='gray')
            axes[0][c].axis('off')
            axes[0][c].set_title(str(ocr1.get_number(image))+ocr1.get_sign(image))
        
        for c, image in enumerate(images_r):
            #print(c)
            axes[1][c].imshow(image, cmap='gray')
            axes[1][c].axis('off')
            axes[1][c].set_title(str(ocr1.get_number(image))+ocr1.get_sign(image))
        plt.show()
    # When everything done, release the video capture object

    print(get_eqn_result("2+2/2*2="))

    while cv.waitKey(300) != ord("q"):
        time.sleep(1)
    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
    print("Done")