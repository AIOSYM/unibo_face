import cv2
import numpy as np
import face_recognition

# my module
import utils 
from inception_blocks_v2 import faceRecoModel

INPUT_SHAPE = (3, 96, 96)
USER_LIST = ['david', 'neruneru']

def initModel():
    model = faceRecoModel(input_shape=INPUT_SHAPE)
    utils.load_weights_from_FaceNet(model)
    return model

def initDatabase(model):
    database = {}
    database[USER_LIST[0]] = utils.img_to_encoding('images/david.jpg', model)
    database[USER_LIST[1]] = utils.img_to_encoding('images/neru.JPG', model)
    return database

def localize_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    return face_locations

def find_ROIs(frame, face_locations, drawBBox=True):
    ROI_LIST = []
    for top, right, bottom, left in face_locations:
        width = right - left
        height = bottom - top
        width_segment =  width // 10
        height_segment = height // 10
        ROI_location = (top, right, bottom, left, width_segment, height_segment)
        ROI = frame[top:bottom, left:right, :]
        ROI = cv2.resize(ROI, (96,96), interpolation = cv2.INTER_AREA)
        ROI_LIST.append(ROI)

        if not drawBBox:
            continue
        utils.drawBBox(frame, ROI_location)

    return ROI_LIST

def identify_face(ROI, database, model):
    MIN_DIST = 100
    encoding = utils.roi_to_encoding(ROI, model)

    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding-db_enc)
        if dist < MIN_DIST:
            MIN_DIST = dist
            identity = name

    if MIN_DIST > 0.65:
        print("Not in the database." + str(MIN_DIST))
    else:
        print ("it's " + str(identity) + ", the distance is " + str(MIN_DIST))
        
    return identity, MIN_DIST

def process_frame(frame, database, model):
    face_locations = localize_faces(frame)
    ROIs = find_ROIs(frame, face_locations)
    for ROI in ROIs:
        identify_face(ROI, database, model)




