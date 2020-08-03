
#################################
#Import Essential Libraries
#################################
#import age_gender_detector
import dlib
from imutils import face_utils
import numpy as np
import imutils
import cv2
import math
from deepface import DeepFace
#################################
#Define Most Useful Functions:
#################################

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
GOLDEN_VALUE = 1.61803398875
COEFFICIENT = 2.38196601125



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


golden_ratio_v = list()
golden_ratio_h = list()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


#################################
#Load the input image, resize it, and convert it to grayscale
#################################

loc = "example.png"
image = cv2.imread(loc)

#ALL ABOUT FACE
attributes = ['age', 'gender' , 'race' , 'emotion']
analysis = DeepFace.analyze(loc,attributes)

print(analysis['age'],analysis['gender'],analysis['dominant_race'],analysis['dominant_emotion'])

#RESIZE AND CONVERT TO GRAYSCALE IMAGE
image = imutils.resize(image, width=512)
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

right_pupil_to_center_of_lips = center_of_right_pupil[1] - mouth_center[1]
#print(right_pupil_to_center_of_lips)
left_pupil_to_center_of_lips = center_of_left_pupil[1] - mouth_center[1]
center_of_lips_to_chin = mouth_center[1] - shape[8][1]

V1 = abs(right_pupil_to_center_of_lips/center_of_lips_to_chin)
golden_ratio_v.append( V1 )


#################################
#V2: Center of pupils ,Nose at nostrils, Bottom of chin
#################################
(nose_at_nostrils_x_right,_) = (shape[35]+shape[42])//2
nose_at_nostrils_right = (nose_at_nostrils_x_right,shape[35][1])
(nose_at_nostrils_x_left,_) = (shape[31]+shape[39])//2
nose_at_nostrils_left = (nose_at_nostrils_x_left,shape[31][1])
cv2.circle(image, nose_at_nostrils_right, 2, (255,255,255) , -1)
cv2.circle(image, nose_at_nostrils_left, 2, (255,255,255) , -1)
right_nose_nostrils_to_right_pupil = nose_at_nostrils_right[1] - center_of_right_pupil[1]
left_nose_nostrils_to_left_pupil = nose_at_nostrils_left[1] - center_of_right_pupil[1]
average_of_nostrilses_to_pupils = (right_nose_nostrils_to_right_pupil+left_nose_nostrils_to_left_pupil)/2

chin = shape[8]

right_nose_nostrils_to_chin = nose_at_nostrils_right[1] - chin[1]
left_nose_nostrils_to_chin = nose_at_nostrils_left[1] - chin[1]
average_of_nostrilses_to_chin = (right_nose_nostrils_to_chin+left_nose_nostrils_to_chin)/2

#TEST
#cv2.line(image,(center_of_right_pupil[0],center_of_right_pupil[1]),(center_of_right_pupil[0],nose_at_nostrils_right[1]),(255,120,255),2)
#cv2.line(image,(nose_at_nostrils_right[0],nose_at_nostrils_right[1]),(nose_at_nostrils_right[0],chin[1]),(255,120,255),2)
V2 =  abs(average_of_nostrilses_to_chin/average_of_nostrilses_to_pupils)
golden_ratio_v.append(V2)


#################################
#V3: Center of pupils ,Nose flair top , Nose base
#################################

nose_flair_top_left = ((shape[29]+shape[39])[0]//2 , (shape[29][1]+shape[30][1])//2)
nose_flair_top_right = ((shape[29]+shape[42])[0]//2 , (shape[29][1]+shape[30][1])//2)


cv2.circle(image, nose_flair_top_left, 2, (255,12,98) , -1)
cv2.circle(image, nose_flair_top_right, 2, (255,12,98) , -1)


pupil_to_flair_top = center_of_left_pupil[1] - nose_flair_top_left[1]


left_nose_base = shape[33][1]

flair_to_nose_base = nose_flair_top_left[1] - left_nose_base


V3 = abs(pupil_to_flair_top/flair_to_nose_base)
golden_ratio_v.append(V3)



#####################################################
#V4: Top arc of eyebrows ,Top of eyes ,Bottom of eyes
#####################################################
top_arc_of_eyebrows = shape[19]
top_of_eyes = shape[37]
bottom_of_eyes = shape[41]

#TEST
#cv2.line(image,(top_arc_of_eyebrows[0],top_arc_of_eyebrows[1]),(top_arc_of_eyebrows[0],top_of_eyes[1]),(255,120,255),2)
#cv2.line(image,(top_of_eyes[0],top_of_eyes[1]),(top_of_eyes[0],bottom_of_eyes[1]),(255,120,255),2)

V4 = abs((top_arc_of_eyebrows[1] - top_of_eyes[1]) /(top_of_eyes[1] - bottom_of_eyes[1]))

golden_ratio_v.append(V4)

#######################################################
#V5: Center of pupils ,Nose at nostrils ,Center of lips
#######################################################
V5 = abs(average_of_nostrilses_to_pupils / (nose_at_nostrils_right[1] - mouth_center[1]))

golden_ratio_v.append(V5)


#######################################################
#V6: Top of lips ,Center of lips ,Bottom of lips ,
#######################################################

top_of_lips = shape[50]
bottom_of_lips = shape[57]

#TEST
#cv2.line(image,(top_of_lips[0],top_of_lips[1]),(top_of_lips[0],mouth_center[1]),(255,120,255),2)
#cv2.line(image,(mouth_center[0],mouth_center[1]),(mouth_center[0],bottom_of_lips[1]),(255,120,255),2)

V6 = abs((mouth_center[1] - bottom_of_lips[1]) / (top_of_lips[1] - mouth_center[1]))

golden_ratio_v.append(V6)

#######################################################
#V7: Nose at nostrils ,	Top of lips,Center of lips
#######################################################

V7 = abs( (nose_at_nostrils_left[1] - top_of_lips[1])/(top_of_lips[1] - mouth_center[1]))

golden_ratio_v.append(V7)


#######################################################
#H1: Side of face ,	Inside of near eye , Opposite side of face
#######################################################
side_of_face = shape[0]
inside_of_near_eye = shape[39]
opposite_side_of_face = shape[16]

#TEST
#cv2.line(image,(side_of_face[0],side_of_face[1]),(inside_of_near_eye[0],side_of_face[1]),(255,120,255),2)
#cv2.line(image,(inside_of_near_eye[0],inside_of_near_eye[1]),(opposite_side_of_face[0],inside_of_near_eye[1]),(255,255,120),2)

H1 = abs((inside_of_near_eye[0] - opposite_side_of_face[0]) / (side_of_face[0] - inside_of_near_eye[0]))

golden_ratio_h.append(H1)

#######################################################
#H2: Side of face 	Inside of near eye  Inside of opposite eye (16)
#######################################################
inside_of_opposite_eye = shape[42]
H2 = abs( (side_of_face[0] - inside_of_near_eye[0])/(inside_of_near_eye[0] - inside_of_opposite_eye[0]))
golden_ratio_h.append(H2)


#######################################################
#H3: Center of face , Outside edge of eye ,	Side of face
#######################################################
center_of_face = shape[8]
outside_edge_of_eye = shape[36]
H3 = abs( (center_of_face[0] - outside_edge_of_eye[0])/(outside_edge_of_eye[0] - side_of_face[0]))
golden_ratio_h.append(H3)


#######################################################
#H4: Side of face 	Outside edge of eye	Inside edge of eye
#######################################################
inside_edge_of_eye = shape[39]

#TEST
#cv2.line(image,(side_of_face[0],side_of_face[1]),(outside_edge_of_eye[0],side_of_face[1]),(255,120,255),2)
#cv2.line(image,(outside_edge_of_eye[0],outside_edge_of_eye[1]),(inside_edge_of_eye[0],outside_edge_of_eye[1]),(255,255,120),2)
H4 = abs((side_of_face[0] - outside_edge_of_eye[0])/(outside_edge_of_eye[0] - inside_edge_of_eye[0]))
golden_ratio_h.append(H4)



#######################################################
#H5: Side of face ,Outside of eye brow ,Outside edge of eye
#######################################################
outside_of_eye_brow = shape[17]

#TEST
#cv2.line(image,(side_of_face[0],side_of_face[1]),(outside_of_eye_brow[0],side_of_face[1]),(255,120,255),2)
#cv2.line(image,(outside_of_eye_brow[0],outside_of_eye_brow[1]),(outside_edge_of_eye[0],outside_of_eye_brow[1]),(255,255,120),2)


H5 = abs((outside_of_eye_brow[0] - outside_edge_of_eye[0])/(side_of_face[0] - outside_of_eye_brow[0]))
golden_ratio_h.append(H5)



#######################################################
#H6: Center of face 	Width of nose 	Width of mouth
#######################################################
width_of_nose = nose_at_nostrils_right[0]
width_of_mouth= shape[54][0]
#cv2.line(image,(center_of_face[0],center_of_face[1]),(nose_at_nostrils_right[0],nose_at_nostrils_right[1]),(255,255,255))
H6 = abs((center_of_face[0] - width_of_nose)/(width_of_nose - width_of_mouth))
golden_ratio_h.append(H6)

#######################################################
#H7:Side of mouth ,Cupid’s bow ,Opposite side of mouth (30)
#######################################################
side_of_mouth = shape[48]
cupids_bow = shape[52]
opposite_side_of_mouth = shape[54]
H7 = abs((side_of_mouth[0] - cupids_bow[0])/(cupids_bow[0] - opposite_side_of_mouth[0]))
golden_ratio_h.append(H7)



# show the output image with the face detections + facial landmarks
#cv2.imshow("Output", image)
#cv2.waitKey(0)w
#cv2.imwrite("result.jpg",image)


#print(golden_ratio_v)
#print(golden_ratio_h)

#print((np.mean(golden_ratio_v)+np.mean(golden_ratio_h))/2)
golden_ratio = golden_ratio_v+golden_ratio_h
print("BEAUTY:",np.mean([(1 - abs(g_vh - GOLDEN_VALUE)/COEFFICIENT)*100 for g_vh in golden_ratio]))
cv2.imshow("Image",image)
cv2.waitKey(0)