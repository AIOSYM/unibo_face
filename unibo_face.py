# Import library
import cv2
import numpy as np
import face_recognition
from utils import img_to_encoding, roi_to_encoding, load_weights, load_weights_from_FaceNet
from inception_blocks_v2 import faceRecoModel

emoji_img = None
model = None
database = None

# List to store coordinate of faces
face_locations = []

def initModel():
    model = faceRecoModel(input_shape=(3, 96, 96))
    #load_weights_from_FaceNet(model)
    return model

def initDatabase():
    database = {}
    database['neruneru'] = img_to_encoding('images/neru.JPG', model)
    database['david'] = img_to_encoding('images/david.jpg', model)
    return database

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Helper function to draw bounding box
def drawSegment(frame, top_left, bottom_right, color=(0,191,255), thickness=5):
    cv2.line(frame, top_left, bottom_right, color, thickness)

def drawEmoji(source_img, emoji_img, x, y):
    SIZE = int(source_img.shape[1] * 0.05)
    emoji_img = cv2.resize(emoji_img, (SIZE, SIZE))
    rows, cols, channels = emoji_img.shape

    if (source_img.shape[1] > x+cols) and (source_img.shape[0]>y+rows):
        emoji_img = cv2.addWeighted(source_img[y:y+rows, x:x+cols], 0.5, emoji_img, 0.5, 0)
        source_img[y:y+rows, x:x+cols] = emoji_img

def drawBBox(frame, locations):
    for top, right, bottom, left in locations:
        width = right - left
        height = bottom - top
        width_segment =  width // 10
        height_segment = height // 10
        ROI = frame[top:bottom, left:right, :]
        ROI = cv2.resize(ROI, (96,96), interpolation = cv2.INTER_AREA)

        drawSegment(frame, (left, top), (left+width_segment, top))
        drawSegment(frame, (left, top), (left, top+height_segment))
        drawSegment(frame, (left, bottom), (left+width_segment, bottom))
        drawSegment(frame, (left, bottom), (left, bottom-height_segment))
        
        drawSegment(frame, (right, top), (right-width_segment, top))
        drawSegment(frame, (right, top), (right, top+height_segment))
        drawSegment(frame, (right, bottom), (right-width_segment, bottom))
        drawSegment(frame, (right, bottom), (right, bottom-height_segment))

        drawEmoji(frame, emoji_img, right, top)

        identify_face(ROI, database, model)

def identify_face(roi, database, model):
    encoding = roi_to_encoding(roi, model)
    min_dist = 100
    
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding-db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database." + str(min_dist))
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity

# Read and process the frame       
def processFrame(CAP):
    ret, frame = CAP.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.resize(rgb_frame, (640, 360))
    
    face_locations = face_recognition.face_locations(rgb_frame)
    drawBBox(rgb_frame, face_locations)
    
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

def main():
    global emoji_img, model, database
    emoji_img = load_image('media/man_emoji.png')
    model = initModel()
    database = initDatabase()

    # Open camera
    CAP = cv2.VideoCapture(0)

    while True:
        frame = processFrame(CAP)
        cv2.imshow("Display", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    CAP.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()