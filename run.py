import subprocess
import cv2
import os
import HandTrackingModule as htm


def mainFunction():

    cap = cv2.VideoCapture(0)
    wCam, hCam = 1400, 700

    folderPath = "FingerImages"
    myList = os.listdir(folderPath)

    cap.set(3, wCam)
    cap.set(4, hCam)

    overlayList = []
    for imPath in myList:
        image = cv2.imread(f"{folderPath}/{imPath}")
        overlayList.append(image)

    detector = htm.handDetector(detectionCon=0.75)
    tipIds = [4, 8, 12, 16, 20]

    # To store the last detected gesture (number of fingers)
    lastGesture = None

    while True:

        success, img = cap.read()
        img = detector.findHands(img)
        img = cv2.flip(img, 1)

        lmList, bbox = detector.findPosition(img, draw=False)

        cv2.putText(
            img,
            "Welcome To AI World House",
            (45, 335),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 0),
            3,
        )
        cv2.putText(
            img,
            "1. Mouse ",
            (45, 375),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 0),
            3,
        )

        cv2.putText(
            img,
            "2. Painter ",
            (45, 415),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 0),
            3,
        )

        cv2.putText(
            img,
            "3. Volume ",
            (45, 455),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 0),
            3,
        )

        cv2.putText(
            img,
            "4. Exit ",
            (45, 495),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 0),
            3,
        )

        if len(lmList) != 0:
            fingers = []
            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # Other four fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)
            print("Total Fingers:", totalFingers)

            # Only trigger a subprocess if the gesture has changed
            if totalFingers != lastGesture:
                lastGesture = totalFingers
                if totalFingers == 1:
                    try:
                        cap.release()
                        cv2.destroyAllWindows()
                        subprocess.Popen(["python", "AIVirtualMouse.py"])
                    except Exception as e:
                        print(f"Error running AIVirtualMouse.py: {e}")
                elif totalFingers == 2:
                    try:
                        cap.release()
                        cv2.destroyAllWindows()
                        subprocess.Popen(["python", "VirtualPainter.py"])

                    except Exception as e:
                        print(f"Error running VirtualPainter.py: {e}")
                elif totalFingers == 3:
                    try:
                        cap.release()
                        cv2.destroyAllWindows()
                        subprocess.Popen(["python", "VolumeHandControl.py"])

                    except Exception as e:
                        print(f"Error running VolumeHandControl.py: {e}")
                elif totalFingers == 4:

                    cap.release()
                    cv2.destroyAllWindows()

                else:
                    print("Invalid gesture: show 1, 2, or 3 fingers.")

            # display overlay image if available for the given finger count

            if totalFingers > 0 and totalFingers <= len(overlayList):
                h, w, c = overlayList[totalFingers - 1].shape
                img[0:h, 0:w] = overlayList[totalFingers - 1]

            # draw a rectangle and display the number of fingers
            cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                img,
                str(totalFingers),
                (45, 375),
                cv2.FONT_HERSHEY_PLAIN,
                10,
                (255, 0, 0),
                25,
            )

        cv2.imshow("AI Finger Counter", img)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break

    cap.release()
    cv2.destroyAllWindows()


mainFunction()
