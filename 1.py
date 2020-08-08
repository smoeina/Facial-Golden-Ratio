
#################################
#Import Essential Libraries
#################################
#import age_gender_detector
import dlib
from imutils import face_utils
import numpy as np
import imutils
import argparse
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

FIRST_COLOR = (68, 5, 7)
SECOND_COLOR = (64, 194, 250)
THIRD_COLOR = (112, 32, 230)



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
predictor = dlib.shape_predictor("code/shape_predictor_68_face_landmarks.dat")


#################################
#Load the input image, resize it, and convert it to grayscale
#################################


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

the_path = "uploads/"+args["image"].split("/")[1]+"/"


# image = cv2.imread("example.png")
# #image = cv2.imread("amber2.jpg")
# #image = cv2.imread("exxample.jpg")
# #image = cv2.imread("hasan.jpg")
# #image = cv2.imread("hanx.jpeg")
# image = cv2.imread("dicaprio.jpeg")
# #image = cv2.imread("brad.jpg")
# #image = cv2.imread("brad2.jpg")
# #image = cv2.imread("me.jpg")
# #image = cv2.imread("mamud.jpg")
image = cv2.imread(args["image"])
# #image = cv2.imread("tnzshownm.jpg")
# #image = cv2.imread("jan.jpg")
# #image = cv2.imread("emam.jpg")
# #image = cv2.imread("najm.jpg")
# #image = cv2.imread("trump.jpg")

#ALL ABOUT FACE
attributes = ['age', 'gender' , 'race' , 'emotion']
analysis = DeepFace.analyze(image,attributes)

print("AGE:",int(analysis['age'])
      ,", GENDER:",analysis['gender']
      ,", RACE:",analysis['dominant_race']
      ,", EMOTION:",analysis['dominant_emotion']
      ,",")

#RESIZE AND CONVERT TO GRAYSCALE IMAGE
image = imutils.resize(image, width=2512)
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
    cv2.rectangle(image, (x, y), (x + w, y + h), THIRD_COLOR, 2)
    image_backup = image.copy()





    # show the face number
    #cv2.putText(image, "Amber Ziba #{}".format(i + 1), (x - 10, y - 10),
#cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x1, y1) in shape:
    	cv2.circle(image, (x1, y1), 1, (0, 0, 255), -1)

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


#TEST
v1 = image.copy()
cv2.line(v1,(center_of_left_pupil[0],center_of_left_pupil[1]),(center_of_left_pupil[0],mouth_center[1]),FIRST_COLOR,2)
cv2.line(v1,(mouth_center[0],mouth_center[1]),(mouth_center[0],shape[8][1]),SECOND_COLOR,2)
cv2.putText(v1, "RATIO {:.2f}".format(V1), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
v1 = v1[y:y+h,x:x+w]
cv2.imwrite(the_path+'/v1.jpg',v1)


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


#TEST
v2 = image.copy()
cv2.line(v2,(center_of_left_pupil[0],center_of_left_pupil[1]),(center_of_left_pupil[0],nose_at_nostrils_left[1]),FIRST_COLOR,2)
cv2.line(v2,(nose_at_nostrils_left[0],nose_at_nostrils_left[1]),(nose_at_nostrils_left[0],chin[1]),SECOND_COLOR,2)
cv2.putText(v2, "RATIO {:.2f}".format(V2), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
v2 = v2[y:y+h,x:x+w]
cv2.imwrite(the_path+'/v2.jpg',v2)


#################################
#V3: Center of pupils ,Nose flair top , Nose base
#################################

nose_flair_top_left = ((shape[29]+shape[39])[0]//2 , (shape[29][1]+shape[30][1])//2)
nose_flair_top_right = ((shape[29]+shape[42])[0]//2 , (shape[29][1]+shape[30][1])//2)


cv2.circle(image, nose_flair_top_left, 2, (255,12,98) , -1)
cv2.circle(image, nose_flair_top_right, 2, (255,12,98) , -1)


pupil_to_flair_top = center_of_left_pupil[1] - nose_flair_top_left[1]


left_nose_base = shape[33]

flair_to_nose_base = nose_flair_top_left[1] - left_nose_base[1]


V3 = abs(pupil_to_flair_top/flair_to_nose_base)
golden_ratio_v.append(V3)

#TEST
v3 = image.copy()
cv2.line(v3,(center_of_left_pupil[0],center_of_left_pupil[1]),(center_of_left_pupil[0],nose_flair_top_left[1]),FIRST_COLOR,2)
cv2.line(v3,(nose_flair_top_left[0],nose_flair_top_left[1]),(nose_flair_top_left[0],left_nose_base[1]),SECOND_COLOR,2)
cv2.putText(v3, "RATIO {:.2f}".format(V3), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
v3 = v3[y:y+h,x:x+w]
cv2.imwrite(the_path+'/v3.jpg',v3)

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

#TEST
v4 = image.copy()
cv2.line(v4,(top_arc_of_eyebrows[0],top_arc_of_eyebrows[1]),(top_arc_of_eyebrows[0],top_of_eyes[1]),FIRST_COLOR,2)
cv2.line(v4,(top_of_eyes[0],top_of_eyes[1]),(top_of_eyes[0],bottom_of_eyes[1]),SECOND_COLOR,2)
cv2.putText(v4, "RATIO {:.2f}".format(V4), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
v4 = v4[y:y+h,x:x+w]
cv2.imwrite(the_path+'/v4.jpg',v4)


#######################################################
#V5: Center of pupils ,Nose at nostrils ,Center of lips
#######################################################
V5 = abs(average_of_nostrilses_to_pupils / (nose_at_nostrils_right[1] - mouth_center[1]))

golden_ratio_v.append(V5)

#TEST
v5 = image.copy()
cv2.line(v5,(center_of_left_pupil[0],center_of_left_pupil[1]),(center_of_left_pupil[0],nose_at_nostrils_left[1]),FIRST_COLOR,2)
cv2.line(v5,(nose_at_nostrils_left[0],nose_at_nostrils_left[1]),(nose_at_nostrils_left[0],mouth_center[1]),SECOND_COLOR,2)
cv2.putText(v5, "RATIO {:.2f}".format(V5), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
v5 = v5[y:y+h,x:x+w]
cv2.imwrite(the_path+'/v5.jpg',v5)



#######################################################
#V6: Top of lips ,Center of lips ,Bottom of lips ,
#######################################################

top_of_lips = shape[50]
bottom_of_lips = shape[57]


V6 = abs((mouth_center[1] - bottom_of_lips[1]) / (top_of_lips[1] - mouth_center[1]))

golden_ratio_v.append(V6)

#TEST
v6 = image.copy()
cv2.line(v6,(top_of_lips[0],top_of_lips[1]),(top_of_lips[0],mouth_center[1]),FIRST_COLOR,2)
cv2.line(v6,(mouth_center[0],mouth_center[1]),(mouth_center[0],bottom_of_lips[1]),SECOND_COLOR,2)
cv2.putText(v6, "RATIO {:.2f}".format(V6), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
v6 = v6[y:y+h,x:x+w]
cv2.imwrite(the_path+'/v6.jpg',v6)

#######################################################
#V7: Nose at nostrils ,	Top of lips,Center of lips 
#######################################################

V7 = abs( (nose_at_nostrils_left[1] - top_of_lips[1])/(top_of_lips[1] - mouth_center[1]))

golden_ratio_v.append(V7)

#TEST
v7 = image.copy()
cv2.line(v7,(nose_at_nostrils_left[0],nose_at_nostrils_left[1]),(nose_at_nostrils_left[0],top_of_lips[1]),FIRST_COLOR,2)
cv2.line(v7,(top_of_lips[0],top_of_lips[1]),(top_of_lips[0],mouth_center[1]),SECOND_COLOR,2)
cv2.putText(v7, "RATIO {:.2f}".format(V7), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
v7 = v7[y:y+h,x:x+w]
cv2.imwrite(the_path+'/v7.jpg',v7)


#######################################################
#H1: Side of face ,	Inside of near eye , Opposite side of face
#######################################################
side_of_face = shape[0]
inside_of_near_eye = shape[39]
opposite_side_of_face = shape[16]


H1 = abs((inside_of_near_eye[0] - opposite_side_of_face[0]) / (side_of_face[0] - inside_of_near_eye[0]))

golden_ratio_h.append(H1)

#TEST
h1 = image.copy()
cv2.line(h1,(side_of_face[0],side_of_face[1]),(inside_of_near_eye[0],side_of_face[1]),FIRST_COLOR,2)
cv2.line(h1,(inside_of_near_eye[0],inside_of_near_eye[1]),(opposite_side_of_face[0],inside_of_near_eye[1]),SECOND_COLOR,2)
cv2.putText(h1, "RATIO {:.2f}".format(H1), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
h1 = h1[y:y+h,x:x+w]
cv2.imwrite(the_path+'/h1.jpg',h1)

#######################################################
#H2: Side of face 	Inside of near eye  Inside of opposite eye (16)
#######################################################
inside_of_opposite_eye = shape[42]
H2 = abs( (side_of_face[0] - inside_of_near_eye[0])/(inside_of_near_eye[0] - inside_of_opposite_eye[0]))
golden_ratio_h.append(H2)


#TEST
h2 = image.copy()
cv2.line(h2,(side_of_face[0],side_of_face[1]),(inside_of_near_eye[0],side_of_face[1]),FIRST_COLOR,2)
cv2.line(h2,(inside_of_near_eye[0],inside_of_near_eye[1]),(inside_of_opposite_eye[0],inside_of_near_eye[1]),SECOND_COLOR,2)
cv2.putText(h2, "RATIO {:.2f}".format(H1), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
h2 = h2[y:y+h,x:x+w]
cv2.imwrite(the_path+'/h2.jpg',h2)


#######################################################
#H3: Center of face , Outside edge of eye ,	Side of face
#######################################################
#center_of_face = shape[8]
center_of_face = shape[30]
outside_edge_of_eye = shape[36]
H3 = abs( (center_of_face[0] - outside_edge_of_eye[0])/(outside_edge_of_eye[0] - side_of_face[0]))
golden_ratio_h.append(H3)

#TEST
h3 = image.copy()
cv2.line(h3,(center_of_face[0],center_of_face[1]),(outside_edge_of_eye[0],center_of_face[1]),FIRST_COLOR,2)
cv2.line(h3,(outside_edge_of_eye[0],outside_edge_of_eye[1]),(side_of_face[0],outside_edge_of_eye[1]),SECOND_COLOR,2)
cv2.putText(h3, "RATIO {:.2f}".format(H1), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
h3 = h3[y:y+h,x:x+w]
cv2.imwrite(the_path+'/h3.jpg',h3)


#######################################################
#H4: Side of face 	Outside edge of eye	Inside edge of eye
#######################################################
inside_edge_of_eye = shape[39]

#TEST
#cv2.line(image,(side_of_face[0],side_of_face[1]),(outside_edge_of_eye[0],side_of_face[1]),(255,120,255),2)
#cv2.line(image,(outside_edge_of_eye[0],outside_edge_of_eye[1]),(inside_edge_of_eye[0],outside_edge_of_eye[1]),(255,255,120),2)
H4 = abs((side_of_face[0] - outside_edge_of_eye[0])/(outside_edge_of_eye[0] - inside_edge_of_eye[0]))
golden_ratio_h.append(H4)

#TEST
h4 = image.copy()
cv2.line(h4,(side_of_face[0],side_of_face[1]),(outside_edge_of_eye[0],side_of_face[1]),FIRST_COLOR,2)
cv2.line(h4,(outside_edge_of_eye[0],outside_edge_of_eye[1]),(inside_edge_of_eye[0],outside_edge_of_eye[1]),SECOND_COLOR,2)
cv2.putText(h4, "RATIO {:.2f}".format(H4), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
h4 = h4[y:y+h,x:x+w]
cv2.imwrite(the_path+'/h4.jpg',h4)

#######################################################
#H5: Side of face ,Outside of eye brow ,Outside edge of eye
#######################################################
outside_of_eye_brow = shape[17]

#TEST
#cv2.line(image,(side_of_face[0],side_of_face[1]),(outside_of_eye_brow[0],side_of_face[1]),(255,120,255),2)
#cv2.line(image,(outside_of_eye_brow[0],outside_of_eye_brow[1]),(outside_edge_of_eye[0],outside_of_eye_brow[1]),(255,255,120),2)


H5 = abs((outside_of_eye_brow[0] - outside_edge_of_eye[0])/(side_of_face[0] - outside_of_eye_brow[0]))
golden_ratio_h.append(H5)


#TEST
h5 = image.copy()
cv2.line(h5,(outside_of_eye_brow[0],outside_of_eye_brow[1]),(side_of_face[0],outside_of_eye_brow[1]),FIRST_COLOR,2)
cv2.line(h5,(side_of_face[0],side_of_face[1]),(outside_of_eye_brow[0],side_of_face[1]),SECOND_COLOR,2)
cv2.putText(h5, "RATIO {:.2f}".format(H5), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
h5 = h5[y:y+h,x:x+w]
cv2.imwrite(the_path+'/h5.jpg',h5)


#######################################################
#H6: Center of face 	Width of nose 	Width of mouth
#######################################################
width_of_nose = nose_at_nostrils_right
width_of_mouth= shape[54]
#cv2.line(image,(center_of_face[0],center_of_face[1]),(nose_at_nostrils_right[0],nose_at_nostrils_right[1]),(255,255,255))
H6 = abs((center_of_face[0] - width_of_nose[0])/(width_of_nose[0] - width_of_mouth[0]))
golden_ratio_h.append(H6)

#TEST
h6 = image.copy()
cv2.line(h6,(center_of_face[0],center_of_face[1]),(width_of_nose[1],center_of_face[1]),FIRST_COLOR,2)
cv2.line(h6,(width_of_nose[0],width_of_nose[1]),(width_of_mouth[0],width_of_nose[1]),SECOND_COLOR,2)
cv2.putText(h6, "RATIO {:.2f}".format(H6), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
h6 = h6[y:y+h,x:x+w]
cv2.imwrite(the_path+'/h6.jpg',h6)


#######################################################

#######################################################
side_of_mouth = shape[48]
cupids_bow = shape[52]
opposite_side_of_mouth = shape[54]
H7 = abs((side_of_mouth[0] - cupids_bow[0])/(cupids_bow[0] - opposite_side_of_mouth[0]))
golden_ratio_h.append(H7)


#TEST
h7 = image.copy()
cv2.line(h7,(side_of_mouth[0],side_of_mouth[1]),(cupids_bow[0],side_of_mouth[1]),FIRST_COLOR,2)
cv2.line(h7,(cupids_bow[0],cupids_bow[1]),(opposite_side_of_mouth[0],cupids_bow[1]),SECOND_COLOR,2)
cv2.putText(h7, "RATIO {:.2f}".format(H7), (x + 200, y + 200),
    cv2.FONT_HERSHEY_SIMPLEX, 5, THIRD_COLOR, 5)
h7 = h7[y:y+h,x:x+w]
cv2.imwrite(the_path+'/h7.jpg',h7)



# show the output image with the face detections + facial landmarks
##cv2.imshow("Output", image)
#cv2.waitKey(0)w
#cv2.imwrite("result.jpg",image)


#print(golden_ratio_v)
#print(golden_ratio_h)

#print((np.mean(golden_ratio_v)+np.mean(golden_ratio_h))/2)
golden_ratio = golden_ratio_v+golden_ratio_h
print("BEAUTY:{:.2f}%".format(np.mean([(1 - abs(g_vh - GOLDEN_VALUE)/COEFFICIENT)*100 for g_vh in golden_ratio])))
#cv2.imshow("Image",image)
cv2.waitKey(0)
