from enum import nonmember

import cv2
import numpy as np

#-------------------
CameraIndex = 1
Scale = 1
Step = 8
TurbTreshold = 2
Counter = 0
#-------------------

def init_camera():
    cap = cv2.VideoCapture(CameraIndex)
    cap.set(cv2.CAP_PROP_FPS, 60)
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Camera niet verbonden")
        return None
    return cap


def process(cap, prev_frame=None, ):
    global Counter

    ret,frame = cap.read()
    if not ret:
        print("ERROR: Camera niet verbonden")
        return None

    Counter +=1
    if Counter < 1:
        return prev_frame
    Counter = 0

    #frame = cv2.resize(frame, (0,0), fx=Scale, fy=Scale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is None:
        return gray

    #optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame,gray, None,
        pyr_scale=0.5,
        levels=3,
        winsize = 15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    #vector visualisatie
    h, w = gray.shape
    vis = frame.copy()
    for y in range(0, h, Step):
        for x in range(0, w, Step):
            fx, fy = flow[y,x]
            cv2.arrowedLine(
                vis,
                (x,y),
                (int(x + fx*5),int(y + fy*5)),
                (0,255,0),
                1,
                tipLength=0.3
            )

    #turbulentie detectie
    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
    #turbulentie = grote vatiatie in magnitude
    turbulence_mask = magnitude > TurbTreshold
    #maak rood overlay voor turbulentie
    #overlay = vis.copy()
    #overlay[turbulence_mask] = [0, 0, 255]
    #alpha = 0.4
    #vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

    #display
    cv2.imshow("Windtunnel flow", vis)
    #debug:magnitude tonen
    mag_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    mag_vis = mag_vis.astype(np.uint8)
    cv2.imshow("Flow magnitude", mag_vis)

    return gray

def main():
    print("Wind Tunnel v0.1")
    cap = init_camera()

    prev_frame = None
    global Counter
    Counter = 0
    while True:
        prev_frame = process(cap, prev_frame)

        key = cv2.waitKey(1)
        if key == 27: #esc
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
