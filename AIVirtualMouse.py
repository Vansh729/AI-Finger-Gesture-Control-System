import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui

wCam, hCam = 640, 480
frameR = 100

pTime = 0
smoothening = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

left_click_flag = False
right_click_flag = False
drag_active = False
prev_scroll_y = None

left_click_threshold = 40  # for index+middle finger close (left click)
right_click_threshold = 100  # for index+middle finger spread (right click)
pinch_threshold = 40  # for pinch (thumb + index for drag)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:

        x1, y1 = lmList[8][1:]

        fingers = detector.fingersUp()

        # for scrolling
        if sum(fingers) == 0:

            current_y = lmList[0][2]
            if prev_scroll_y is None:
                prev_scroll_y = current_y
            diff = current_y - prev_scroll_y
            scroll_sensitivity = 2
            if abs(diff) > 10:

                # negative diff scrolls up positive scrolls down
                pyautogui.scroll(-int(diff * scroll_sensitivity))
            prev_scroll_y = current_y

            # reset other gesture flags
            drag_active = False
            left_click_flag = False
            right_click_flag = False

        else:
            prev_scroll_y = None

            # pinch for drag & drop ---

            # Check distance between thumb and index

            pinch_length, img, _ = detector.findDistance(4, 8, img, draw=False)
            if pinch_length < pinch_threshold:

                # pinch detected – initiate or continue drag mode

                if not drag_active:
                    drag_active = True
                    pyautogui.mouseDown()

                # cursor position during drag

                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                autopy.mouse.move(wScr - clocX, clocY)
                plocX, plocY = clocX, clocY

                # reset clicking flags while dragging

                left_click_flag = False
                right_click_flag = False

            else:
                # end drag if pinch is released
                if drag_active:
                    pyautogui.mouseUp()
                    drag_active = False

                # two Fingers (index & middle) for clicking ---

                # index and middle are up

                if fingers[1] == 1 and fingers[2] == 1:

                    # measure distance between index and middle fingers

                    imm_length, img, _ = detector.findDistance(8, 12, img, draw=False)
                    if imm_length < left_click_threshold:

                        # left click
                        if not left_click_flag:
                            autopy.mouse.click()
                            left_click_flag = True
                    elif imm_length > right_click_threshold:
                        # right click
                        if not right_click_flag:
                            pyautogui.click(button="right")
                            right_click_flag = True
                    else:
                        left_click_flag = False
                        right_click_flag = False
                else:
                    # reset click flags
                    left_click_flag = False
                    right_click_flag = False

                    # single index finger cursor move

                    if fingers[1] == 1 and fingers[2] == 0:
                        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening
                        autopy.mouse.move(wScr - clocX, clocY)
                        plocX, plocY = clocX, clocY

    # calculate FPS

    cTime = time.time()
    fps = int(1 / (cTime - pTime)) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {fps}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("AI Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord(" "):
        break

cap.release()
cv2.destroyAllWindows()
