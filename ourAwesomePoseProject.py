import cv2
import time
import poseModule as poseMod


cap = cv2.VideoCapture('poseVideos/3.mp4')
pTime=0
detector = poseMod.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)

    # if len(lmList) != 0:
    #     cv2.circle(img, (lmList[20][1], lmList[20][2]), 10, (155, 20, 230), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 30, 20), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)