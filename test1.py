import cv2
import numpy as np

# -------------------
CameraIndex = 1
Step = 10              # minder pijlen = duidelijker
FrameSkip = 3          # grotere beweging zichtbaar maken
Alpha = 0.7            # temporal smoothing
# -------------------

def init_camera():
    cap = cv2.VideoCapture(CameraIndex)
    cap.set(cv2.CAP_PROP_FPS, 60)

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Camera niet verbonden")
        return None
    return cap


def process(cap, prev_frame, prev_mag):
    ret, frame = cap.read()
    if not ret:
        return prev_frame, prev_mag

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 🔧 blur input → minder noise
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    if prev_frame is None:
        return gray, prev_mag

    # 🔥 optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, gray, None,
        pyr_scale=0.5,
        levels=4,
        winsize=9,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0
    )

    # 🔧 flow smoothing (belangrijk)
    flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (9, 9), 0)
    flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (9, 9), 0)

    # 🔥 magnitude & angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 🔧 temporal smoothing (super belangrijk)
    if prev_mag is None:
        smooth_mag = magnitude
    else:
        smooth_mag = Alpha * prev_mag + (1 - Alpha) * magnitude

    # 🔧 threshold → noise weg
    mean_mag = np.mean(smooth_mag)
    mask = smooth_mag > (mean_mag * 1.5)

    # -------------------
    # 🎨 HSV FLOW VISUALISATIE (beste!)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(smooth_mag, None, 0, 255, cv2.NORM_MINMAX)

    flow_vis = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    cv2.imshow("Flow HSV", flow_vis)

    # -------------------
    # 🧭 PIJLEN (alleen sterke flow)
    vis = frame.copy()
    h, w = gray.shape

    for y in range(0, h, Step):
        for x in range(0, w, Step):
            if mask[y, x]:
                fx, fy = flow[y, x]
                cv2.arrowedLine(
                    vis,
                    (x, y),
                    (int(x + fx * 5), int(y + fy * 5)),
                    (0, 255, 0),
                    1,
                    tipLength=0.2
                )

    cv2.imshow("Flow vectors", vis)

    # -------------------
    # 🔥 magnitude debug (cleaner)
    mag_vis = cv2.normalize(smooth_mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_vis = mag_vis.astype(np.uint8)
    mag_vis = cv2.GaussianBlur(mag_vis, (9, 9), 0)

    cv2.imshow("Flow magnitude", mag_vis)

    return gray, smooth_mag


def main():
    print("Wind Tunnel v2.0")

    cap = init_camera()
    prev_frame = None
    prev_mag = None

    frame_count = 0

    while True:
        frame_count += 1

        # 🔥 frame skipping → grotere beweging
        if frame_count % FrameSkip != 0:
            cap.read()
            continue

        prev_frame, prev_mag = process(cap, prev_frame, prev_mag)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()