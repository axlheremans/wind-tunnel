import cv2
import numpy as np

# -------------------
CameraIndex = 1
Step = 16
FrameSkip = 3

Alpha_flow = 0.45
Alpha_mag = 0.80
# -------------------


def init_camera():
    cap = cv2.VideoCapture(CameraIndex)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Camera niet verbonden")
        return None
    return cap


def process(cap, prev_frame, prev_flow, prev_mag):
    ret, frame = cap.read()
    if not ret:
        return prev_frame, prev_flow, prev_mag

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if prev_frame is None:
        return gray, prev_flow, prev_mag

    # -------------------
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, gray, None,
        pyr_scale=0.5,
        levels=4,
        winsize=9,
        iterations=4,
        poly_n=7,
        poly_sigma=1.3,
        flags=0
    )

    if prev_flow is None:
        smooth_flow = flow
    else:
        smooth_flow = prev_flow + Alpha_flow * (flow - prev_flow)

    magnitude, angle = cv2.cartToPolar(
        smooth_flow[..., 0],
        smooth_flow[..., 1]
    )

    # =====================================================
    # 🔥 MAGNITUDE — SHARPER VERSION
    # =====================================================

    mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    mag = mag.astype(np.uint8)

    # 🔥 lichte smoothing (minder dan before)
    mag = cv2.GaussianBlur(mag, (3, 3), 0)

    # 🔥 CONTRAST BOOST (CLAHE = veel scherper)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    mag = clahe.apply(mag)

    cv2.imshow("Flow magnitude", mag)

    # =====================================================
    # 🔥 HSV — SHARPER V CHANNEL
    # =====================================================

    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    hsv[..., 0] = angle * 180 / np.pi / 2

    hsv_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv_mag = hsv_mag.astype(np.uint8)

    # 🔥 minder blur = meer detail
    hsv_mag = cv2.GaussianBlur(hsv_mag, (3, 3), 0)

    hsv[..., 2] = hsv_mag

    flow_vis = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    cv2.imshow("Flow HSV", flow_vis)

    # -------------------
    # PIJLEN (unchanged)
    vis = frame.copy()
    h, w = gray.shape

    threshold = 0.3

    for y in range(0, h, Step):
        for x in range(0, w, Step):
            fx, fy = smooth_flow[y, x]

            if fx * fx + fy * fy > threshold:
                cv2.arrowedLine(
                    vis,
                    (x, y),
                    (int(x + fx * 8), int(y + fy * 8)),
                    (0, 255, 0),
                    1,
                    tipLength=0.25
                )

    cv2.imshow("Flow vectors", vis)

    return gray, smooth_flow, mag


def main():
    print("Wind Tunnel v3.6 (sharp magnitude + HSV)")

    cap = init_camera()

    prev_frame = None
    prev_flow = None
    prev_mag = None

    frame_count = 0

    while True:
        frame_count += 1

        if frame_count % FrameSkip != 0:
            cap.read()
            continue

        prev_frame, prev_flow, prev_mag = process(
            cap,
            prev_frame,
            prev_flow,
            prev_mag
        )

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
