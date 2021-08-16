import cv2
import mediapipe as mp
import time


class poseDetector():

    def __init__(self, mode = False, upBody = False, smooth = True, 
                detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, 
                                    self.detectionCon, self.trackCon)

    def findPose(self, img, draw = True):
        
        #convertendo de BGR para RGB, entendível por mediapipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB) #detecção

        if self.results.pose_landmarks and draw: #se algum ponto foi detectado
            #desenhando pontos encontrados na imagem:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
    
        return img

    def findPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            #enumerate retorna cada item da lista com seu índice
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 20, 255), cv2.FILLED)

        return lmList



def main():
    cap = cv2.VideoCapture('poseVideos/1.mp4')
    pTime=0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
           cv2.circle(img, (lmList[20][1], lmList[20][2]), 10, (155, 20, 230), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 30, 20), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()