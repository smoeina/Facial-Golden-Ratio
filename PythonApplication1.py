
#################################
#Import Essential Libraries
#################################
import dlib
from imutils import face_utils
import numpy as np
import imutils
import cv2
import math
#################################
#Define Most Useful Functions:
#################################

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

def distance(pt1,pt2):
  return math.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)


def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

#################################
#Define Most Useful Functions:
#################################


golden_ratio = list()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


#################################
#Load the input image, resize it, and convert it to grayscale
#################################

image = cv2.imread("amber2.jpg")
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
	# rotate to good vision
	left_eye = extract_left_eye_center(shape)
	right_eye = extract_right_eye_center(shape)
	M = get_rotation_matrix(left_eye, right_eye)
	image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
	gray = cv2.warpAffine(gray, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)


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
    cv2.putText(image, "Amber Ziba #{}".format(i + 1), (x - 10, y - 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
    	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

#################################
#V1: Center of pupils ,Center of lips, Bottom of chin
#################################

center_of_left_pupil = ((shape[43]+shape[44])[0]//2 , (shape[43]+shape[47])[1]//2)
cv2.circle(image, center_of_left_pupil, 4, (255, 0, 255), -1)
center_of_right_pupil = ((shape[37]+shape[38])[0]//2 , (shape[37]+shape[41])[1]//2)
cv2.circle(image,center_of_right_pupil, 4, (255, 0, 255), -1)
(x_mouth_center,y_mouth_center) = (shape[63]+shape[67])//2
mouth_center = (x_mouth_center , y_mouth_center)
cv2.circle(image, (x_mouth_center,y_mouth_center), 2, (255,255,0), -1)
right_pupil_to_center_of_lips = distance(center_of_right_pupil,mouth_center)
left_pupil_to_center_of_lips = distance(center_of_left_pupil,mouth_center)
center_of_lips_to_chin = distance(mouth_center,shape[8])

V1 = right_pupil_to_center_of_lips/center_of_lips_to_chin
golden_ratio.append( V1 )


#################################
#V2: Center of pupils ,Nose at nostrils, Bottom of chin
#################################
(nose_at_nostrils_x_left,_) = (shape[35]+shape[42])//2
nose_at_nostrils_left = (nose_at_nostrils_x_left,shape[35][1])
(nose_at_nostrils_x_right,_) = (shape[31]+shape[39])//2
nose_at_nostrils_right = (nose_at_nostrils_x_right,shape[31][1])
cv2.circle(image, nose_at_nostrils_left, 2, (255,12,98) , -1)
cv2.circle(image, nose_at_nostrils_right, 2, (255,12,98) , -1)

right_nose_nostrils_to_right_pupil = distance(nose_at_nostrils_right,center_of_right_pupil)
left_nose_nostrils_to_left_pupil = distance(nose_at_nostrils_left,center_of_right_pupil)
average_of_nostrilses_to_pupils = (right_nose_nostrils_to_right_pupil+left_nose_nostrils_to_left_pupil)/2

chin = shape[8]

right_nose_nostrils_to_chin = distance(nose_at_nostrils_right,chin)
left_nose_nostrils_to_chin = distance(nose_at_nostrils_left,chin)
average_of_nostrilses_to_chin = (right_nose_nostrils_to_chin+left_nose_nostrils_to_chin)/2


V2 = average_of_nostrilses_to_pupils / average_of_nostrilses_to_chin
golden_ratio.append(V2)


#################################
#V3: Center of pupils ,Nose flair top , Nose base
#کج بشه ساپورت نمیشه!
#################################

nose_flair_top_left = ((shape[29]+shape[39])[0]//2 , (shape[29][1]+shape[30][1])//2)
nose_flair_top_right = ((shape[29]+shape[42])[0]//2 , (shape[29][1]+shape[30][1])//2)


cv2.circle(image, nose_flair_top_left, 2, (255,12,98) , -1)
cv2.circle(image, nose_flair_top_right, 2, (255,12,98) , -1)


pupil_to_flair_top_left =distance(center_of_left_pupil,nose_flair_top_left)
pupil_to_flair_top_right =distance(center_of_right_pupil,nose_flair_top_right)

pupil_to_flair_top_avg = (pupil_to_flair_top_left + pupil_to_flair_top_right) /2

left_nose_base = shape[33]
right_nose_base = shape[34]

left_flair_to_nose_base =distance(nose_flair_top_left,left_nose_base)
right_flair_to_nose_base =distance(nose_flair_top_right,right_nose_base)

avg_flair_to_nose_base = (right_flair_to_nose_base + left_flair_to_nose_base)/2

V3 = pupil_to_flair_top_avg/avg_flair_to_nose_base
golden_ratio.append(V3)
# show the output image with the face detections + facial landmarks
#cv2.imshow("Output", image)
#cv2.waitKey(0)
#cv2.imwrite("result.jpg",image)
print(golden_ratio)
cv2.imshow("Image",image)
cv2.waitKey(0)