import cv2
import numpy as np

CameraIndex = 1

def init_camera():
    cap = cv2.VideoCapture(CameraIndex)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 60)

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Camera niet verbonden")
        return None
    return cap


cap = init_camera()
if cap is None:
    exit()

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

step = 15
frame_skip = 2

# 🔥 exponentiële smoothing
smoothed_flow = None
alpha = 0.2   # lager = stabieler, hoger = responsiever

# 🔥 NIEUW: gevoeligheid voor turbulentie (rood)
turbulence_sensitivity = 1.5

while True:
    # frameskip
    for _ in range(frame_skip):
        ret, frame = cap.read()
        if not ret:
            break

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        old_gray, gray,
        None,
        0.5, 4, 20, 3, 5, 1.2, 0
    )

    # lichte blur (niet te sterk!)
    flow = cv2.GaussianBlur(flow, (5, 5), 0)

    # 🔥 exponentiële smoothing
    if smoothed_flow is None:
        smoothed_flow = flow
    else:
        smoothed_flow = (1 - alpha) * smoothed_flow + alpha * flow

    h, w = gray.shape

    kernel_size = 5

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = smoothed_flow[y, x]
            magnitude = np.sqrt(dx ** 2 + dy ** 2)

            # alleen echt stilstaande pixels skippen
            if magnitude < 0.1:
                continue

            # lokale regio
            x1 = max(0, x - kernel_size)
            x2 = min(w, x + kernel_size)
            y1 = max(0, y - kernel_size)
            y2 = min(h, y + kernel_size)

            local_flow = smoothed_flow[y1:y2, x1:x2]

            # 🔥 magnitude component (lange pijlen = meer turbulent)
            local_mag = np.sqrt(local_flow[..., 0] ** 2 + local_flow[..., 1] ** 2)
            avg_mag = np.mean(local_mag)

            # normaliseren (pas deze aan indien nodig)
            mag_norm = min(avg_mag / 2.0, 1.0)

            # 🔥 curvature (hoeveel de flow "buigt")
            angles = np.arctan2(local_flow[..., 1], local_flow[..., 0])
            angle_diff = np.abs(np.diff(angles, axis=0)).mean() + np.abs(np.diff(angles, axis=1)).mean()

            curvature = min(angle_diff, 1.0)

            # 🔥 combineer beide
            t = 0.5 * mag_norm + 0.5 * curvature

            # gevoeligheid
            t = min(t * turbulence_sensitivity, 1.0)

            red = int(255 * t)
            green = int(255 * (1 - t))
            color = (0, green, red)

            # pijltje tekenen
            scale = 2 + magnitude * 1.2

            end_x = int(x + dx * scale)
            end_y = int(y + dy * scale)

            # minimum lengte zodat turbulentie zichtbaar blijft
            min_length = 2
            if abs(end_x - x) < min_length and abs(end_y - y) < min_length:
                end_x = int(x + np.sign(dx) * min_length)
                end_y = int(y + np.sign(dy) * min_length)

            cv2.arrowedLine(
                frame,
                (x, y),
                (end_x, end_y),
                color,
                1,
                tipLength=0.15
            )

    cv2.imshow("Improved Flow", frame)

    old_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()