import cv2
import numpy as np
import time
import PoseModule as pm

# URL ou chemin vers la vidéo de squat
# url = "../images/squat_video4.mp4"
url = "video/squat_video4.mp4"
# url = "http://192.168.11.104:4747/video"

# Ouvrir le flux vidéo
cap = cv2.VideoCapture(url)

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break
    # img = cv2.imread("squatE.jpg")
    # img = cv2.resize(img, (350, 500))
    (h, w) = img.shape[:2]

    # center = (w / 2, h / 2)    
    # M = cv2.getRotationMatrix2D(center, 90, 1.0)
    # img = cv2.warpAffine(img, M, (w, h))
    
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        # Calcul des angles
        back_angle = detector.findAngle(img, 12, 24, 26)
        knee_angle = detector.findAngle(img, 24, 26, 28)

        # Interpoler les valeurs d'angle pour le pourcentage et la barre de progression
        per1 = np.interp(back_angle, (65, 170), (0, 100))
        per2 = np.interp(knee_angle, (65, 165), (0, 100))
        bar = np.interp(back_angle, (70, 170), (650, 100))

        per = (per1 + per2) / 2
        # print("back_angle : " + back_angle)
        print("back_angle : " + str(back_angle))
        # print("knee_angle : " + knee_angle)
        print("knee_angle : " + str(knee_angle))
        print(per)
        print(bar)
        color = (255, 0, 255)
        if 95 <= per <= 100 :
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per <= 5 :
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        # Dessiner la barre de progression
        cv2.rectangle(img, (1100, 100), (w-60, h-20), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (w-60, h-20), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (w-120, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                    color, 4)


        cv2.rectangle(img, (0, 550), (150, 700), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (10, h-30), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 5)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
