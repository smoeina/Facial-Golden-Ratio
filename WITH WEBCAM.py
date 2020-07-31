#################################
# Import Essential Libraries
#################################
import dlib
from imutils import face_utils
import numpy as np
import imutils
import cv2
import math


#################################
# Define Most Useful Functions:
#################################

def distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


#################################
# Define Most Useful Functions:
#################################
golden_ratio = list()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#################################
# Load the input image, resize it, and convert it to grayscale
#################################

cap = cv2.VideoCapture(0)
while True:
#image = cv2.imread("amber.jpg")
    _,image = cap.read()
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "FACE #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        #################################
    # V1: Center of pupils ,Center of lips, Bottom of chin
    #################################

    center_of_left_pupil = ((shape[43] + shape[44])[0] // 2, (shape[43] + shape[47])[1] // 2)
    cv2.circle(image, center_of_left_pupil, 4, (255, 0, 255), -1)
    center_of_right_pupil = ((shape[37] + shape[38])[0] // 2, (shape[37] + shape[41])[1] // 2)
    cv2.circle(image, center_of_right_pupil, 4, (255, 0, 255), -1)
    (x_mouth_center, y_mouth_center) = (shape[63] + shape[67]) // 2
    mouth_center = (x_mouth_center, y_mouth_center)
    cv2.circle(image, (x_mouth_center, y_mouth_center), 2, (255, 255, 0), -1)
    right_pupil_to_center_of_lips = distance(center_of_right_pupil, mouth_center)
    left_pupil_to_center_of_lips = distance(center_of_left_pupil, mouth_center)
    center_of_lips_to_chin = distance(mouth_center, shape[8])

    V1 = right_pupil_to_center_of_lips / center_of_lips_to_chin
    golden_ratio.append(V1)

    #################################
    # V2: Center of pupils ,Nose at nostrils, Bottom of chin
    #################################
    (nose_at_nostrils_x_left, _) = (shape[35] + shape[42]) // 2
    nose_at_nostrils_left = (nose_at_nostrils_x_left, shape[35][1])
    (nose_at_nostrils_x_right, _) = (shape[31] + shape[39]) // 2
    nose_at_nostrils_right = (nose_at_nostrils_x_right, shape[31][1])
    cv2.circle(image, nose_at_nostrils_left, 2, (255, 12, 98), -1)
    cv2.circle(image, nose_at_nostrils_right, 2, (255, 12, 98), -1)

    right_nose_nostrils_to_right_pupil = distance(nose_at_nostrils_right, center_of_right_pupil)
    left_nose_nostrils_to_left_pupil = distance(nose_at_nostrils_left, center_of_right_pupil)
    average_of_nostrilses_to_pupils = (right_nose_nostrils_to_right_pupil + left_nose_nostrils_to_left_pupil) / 2

    chin = shape[8]

    right_nose_nostrils_to_chin = distance(nose_at_nostrils_right, chin)
    left_nose_nostrils_to_chin = distance(nose_at_nostrils_left, chin)
    average_of_nostrilses_to_chin = (right_nose_nostrils_to_chin + left_nose_nostrils_to_chin) / 2

    V2 = average_of_nostrilses_to_pupils / average_of_nostrilses_to_chin
    golden_ratio.append(V2)

    #################################
    # V3: Center of pupils ,Nose flair top , Nose base
    # کج بشه ساپورت نمیشه!
    #################################

    nose_flair_top_left = ((shape[29] + shape[39])[0] // 2, (shape[29][1] + shape[30][1]) // 2)
    nose_flair_top_right = ((shape[29] + shape[42])[0] // 2, (shape[29][1] + shape[30][1]) // 2)

    cv2.circle(image, nose_flair_top_left, 2, (255, 12, 98), -1)
    cv2.circle(image, nose_flair_top_right, 2, (255, 12, 98), -1)

    pupil_to_flair_top_left = distance(center_of_left_pupil, nose_flair_top_left)
    pupil_to_flair_top_right = distance(center_of_right_pupil, nose_flair_top_right)

    pupil_to_flair_top_avg = (pupil_to_flair_top_left + pupil_to_flair_top_right) / 2

    left_nose_base = shape[33]
    right_nose_base = shape[34]

    left_flair_to_nose_base = distance(nose_flair_top_left, left_nose_base)
    right_flair_to_nose_base = distance(nose_flair_top_right, right_nose_base)

    avg_flair_to_nose_base = (right_flair_to_nose_base + left_flair_to_nose_base) / 2

    V3 = pupil_to_flair_top_avg / avg_flair_to_nose_base
    golden_ratio.append(V3)
    # show the output image with the face detections + facial landmarks
    # cv2.imshow("Output", image)
    # cv2.waitKey(0)
    # cv2.imwrite("result.jpg",image)
    print(golden_ratio)
    cv2.imshow("Image", image)
    if cv2.waitKey(1) == 27:
        break
