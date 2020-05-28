import numpy as np
from numpy.linalg import norm
import cv2 as cv
from skimage.measure import inertia_tensor
import matplotlib.pyplot as plt
from main import*


def detect_chars_pos_and_img_ellipse(frame, robot_pos):
    """
    Detect where all the characters are in the frame and returns the list of it.
    Frame is single channel image of the blue and black values (characters on the area).
    """
    # Morphology to make the equal and divide one shape
    kernel = np.zeros((9, 9), np.uint8)
    cv.circle(img=kernel, center=(4, 4), radius=4, color=255, thickness=-1)
    dilated_img = cv.morphologyEx(frame, cv.MORPH_DILATE, kernel, iterations=1)
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
            r = range(3, 60)
            R = range(15, 60)
            robot_pos_threshold = 100
            if int(ma) in r and int(MA) in R and norm((x-robot_pos[0], y-robot_pos[1])) > robot_pos_threshold:
                # Rotate the image to allign the big axis verticaly
                rotation_matrix = cv.getRotationMatrix2D((x, y), angle, 1)
                rotated_img = cv.warpAffine(
                    frame, rotation_matrix, (frame.shape[1], frame.shape[0]))
                # Extract the char image for the rotated image
                extracted_img = rotated_img[int(
                    y-MA/2)-4:int(y+MA/2)+4, int(x-ma/2)-4:int(x+ma/2)+4]
                # Normalize image for the CNN
                normalized_img = normalize_img(extracted_img)
                # Add char image to the list
                chars_img.append(normalized_img)
                # Add position of char to the list
                chars_pos.append((x, y))
    return chars_pos, chars_img

def detect_chars_pos_and_img_ellipse_print(frame, robot_pos, base_image):
    """
    Detect where all the characters are in the frame and returns the list of it.
    Frame is single channel image of the blue and black values (characters on the area).
    """
    # Morphology to make the equal and divide one shape
    kernel = np.zeros((9, 9), np.uint8)
    cv.circle(img=kernel, center=(4, 4), radius=4, color=255, thickness=-1)
    dilated_img = cv.morphologyEx(frame, cv.MORPH_DILATE, kernel, iterations=1)
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
            r = range(3, 60)
            R = range(15, 60)
            robot_pos_threshold = 100
            if int(ma) in r and int(MA) in R and norm((x-robot_pos[0], y-robot_pos[1])) > robot_pos_threshold:
                # Rotate the image to allign the big axis verticaly
                rotation_matrix = cv.getRotationMatrix2D((x, y), angle, 1)
                rotated_img = cv.warpAffine(
                    frame, rotation_matrix, (frame.shape[1], frame.shape[0]))
                # Extract the char image for the rotated image
                extracted_img = rotated_img[int(
                    y-MA/2)-4:int(y+MA/2)+4, int(x-ma/2)-4:int(x+ma/2)+4]
                # Normalize image for the CNN
                normalized_img = normalize_img(extracted_img)
                # Add char image to the list
                chars_img.append(normalized_img)
                # Add position of char to the list
                chars_pos.append((x, y))
                cv.ellipse(base_image, ellipse,(0,255,0),2)
            else:
                cv.ellipse(base_image, ellipse,(255,0,0),2)
    cv.imshow("test", base_image)
    return chars_pos, chars_img

def detect_chars_pos_and_img_print(frame, robot_pos, base_image):
    """
    Detect where all the characters are in the frame and returns the list of it.
    Frame is single channel image of the blue and black values (characters on the area).
    """
    # Morphology to make the equal and divide one shape
    kernel = np.zeros((9, 9), np.uint8)
    cv.circle(img=kernel, center=(4, 4), radius=4, color=255, thickness=-1)
    dilated_img = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel, iterations=1)

    # Find the contours of images
    contours, _ = cv.findContours(
        image=dilated_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    chars_pos = []
    chars_img = []

    for contour in contours:
        # Find bounding box for each contour
        rect = cv.boundingRect(contour)
        x, y, w, h = rect
        # Verify with the box is big enough and not close to the robot
        r = range(2, 60)
        R = range(15, 60)
        robot_pos_threshold = 50
        if np.min([w, h]) in r and np.max([w, h]) in R and norm((x+w/2-robot_pos[0], y+h/2-robot_pos[1])) > robot_pos_threshold:
            # Extract image
            extracted_img = frame[y:y+h, x:x+w]
            # Compute angle of rotation
            mu = inertia_tensor(extracted_img)
            alpha = np.degrees(1/2 *np.arctan2(2*mu[0][1],(mu[1][1]-mu[0][0])))
            # Rotate image to align vertically
            rotation_matrix = cv.getRotationMatrix2D((int(w/2), int(h/2)), alpha, 1)
            rotated_img = cv.warpAffine(
                extracted_img, rotation_matrix, (np.max([extracted_img.shape[1],extracted_img.shape[0]])+5,np.max([extracted_img.shape[1],extracted_img.shape[0]])))
            # Normalize image for the CNN
            normalized_img = normalize_img(rotated_img)
            # Add position of char to the list
            chars_pos.append((x+w/2, y+h/2))
            # Add char image to the list
            chars_img.append(normalized_img)
            cv.rectangle(base_image, rect,(0,255,0),1)
            cv.circle(base_image, center=(int(x+w/2), int(y+h/2)), radius=3, color=(0,255,0), thickness=-1)
        else :
            cv.rectangle(base_image, rect,(255,0,0),1)
    cv.imshow("Bounding box", base_image)
    return chars_pos, chars_img

def main():

    cv.destroyAllWindows()
    cap = cv.VideoCapture("../data/robot_parcours_1.avi")
    r1 = [[10, 10], [170, 80], [170, 80]]  # first red (around 0+ hue)
    r2 = [[170, 10], [170, 80], [170, 80]]  # second red (around 0- hue)
    b1 = [[118, 20], [150, 60], [110, 60]]  # blue
    b2 = [[90, 90], [130, 130], [60, 60]]  # black

    cnn = ocr.DigitNet()
    cnn.load_state_dict(torch.load("./mnist_net.cnn"))

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    if cap.isOpened():
        ret, frame = cap.read()
        #frame = cv.imread("FirstImageMod.jpg")
        
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        chars_img = split(hsv_frame, b1, b2)
        cv.imshow("black image", chars_img)
        
        positions, images = detect_chars_pos_and_img_print(chars_img, (539, 354), frame)
        #positions, images = detect_chars_pos_and_img_ellipse(chars_img, (539, 354))
        #positions, images = detect_chars_pos_and_img_ellipse_print(chars_img, (539, 354), frame)
        chars = determine_chars(images, cnn)
   
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