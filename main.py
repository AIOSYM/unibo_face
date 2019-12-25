import cv2
import numpy as np
import face_recognition

# my module
import utils 
import unibo_face as uf
from inception_blocks_v2 import faceRecoModel

def main():

    model = uf.initModel()
    database = uf.initDatabase(model)

    CAP = cv2.VideoCapture(0)

    while True:
        isRead, frame = CAP.read()
        if isRead:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 360))
            uf.process_frame(frame, database, model)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Display", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    CAP.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()