import sys
import cv2
import numpy as np
import face_recognition

# my module
import utils 
import unibo_face as uf
from inception_blocks_v2 import faceRecoModel

SKIP_FRAMES = 3

def main():

    model = uf.initModel()
    database = uf.initDatabase(model)

    CAP = cv2.VideoCapture(0)
    
    if not CAP.isOpened():
        print('Unable to connect to camera')
        sys.exit()
        
    count = 0
    while True:
        isRead, frame = CAP.read()
        if not isRead:
            print('Unable to read frame')
            sys.exit()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 360))
        
        if (count%SKIP_FRAMES == 0):
            uf.process_frame(frame, database, model)
            
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Display", frame)
        
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    CAP.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()