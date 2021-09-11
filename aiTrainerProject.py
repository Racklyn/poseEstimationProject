import cv2
import numpy as np
import time
import poseModule as pMod

cap = cv2.VideoCapture('aiTrainer/curls.mp4') # Para usar a WebCan, coloque 0

detector = pMod.poseDetector()

count = 0 # number of reps
dir = 0  # 0 = going up; 1 = going down

pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1024, 576)) # 1280, 720
    # img = cv2.imread('aiTrainer/test.jpg')

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    # print(lmList)
    if len(lmList) != 0:
        # Braço direito
        # detector.findAngle(img, 12, 14, 16)

        # Braço esquerdo
        angle = detector.findAngle(img, 11, 13, 15)

        #Convertendo valor do ângulo para porcentagem (0-100)
        per = np.interp(angle, (210, 300), (0,100))
        # print(angle, per)
        bar = np.interp(angle, (210, 300), (530,100))


        color = (255, 0, 255)
        #Check for the dumbbell curls
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 0, 255)
            if dir == 1:
                count += 0.5
                dir = 0

        print(count)

        # Creating bar
        cv2.rectangle(img, (920, 100), (980, 530), color, 3)
        cv2.rectangle(img, (920, int(bar)), (980, 530), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (900, 60), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

        # Draw Curl Count
        cv2.rectangle(img, (0, 366), (210, 576), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(count)}', (40, 530), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 15)
    

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'{int(fps)} fps', (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)