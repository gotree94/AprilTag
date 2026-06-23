import cv2
import json
import sys
from pathlib import Path

VIDEO_PATH = Path(__file__).parent / "cam.mp4"
ROI_CONFIG_PATH = Path(__file__).parent / "roi_config.json"

WINDOW_NAME = "ROI Selection"
rois = []          # list of {"label": str, "x": int, "y": int, "w": int, "h": int}
dragging = False
ix, iy = -1, -1
current_label = ""
label_counter = [1]


def draw_rois(img, rois_list, active_idx=-1):
    display = img.copy()
    for i, roi in enumerate(rois_list):
        color = (0, 255, 0) if i != active_idx else (0, 200, 255)
        cv2.rectangle(display, (roi["x"], roi["y"]),
                      (roi["x"] + roi["w"], roi["y"] + roi["h"]), color, 2)
        label = roi["label"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lx, ly = roi["x"], max(roi["y"] - 10, 20)
        cv2.rectangle(display, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), color, -1)
        cv2.putText(display, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return display


def mouse_callback(event, x, y, flags, param):
    global dragging, ix, iy
    frame = param["frame"]
    display = param["display"]

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        temp = draw_rois(frame, rois)
        cv2.rectangle(temp, (ix, iy), (x, y), (0, 200, 255), 2)
        w, h = x - ix, y - iy
        label = f"Car {label_counter[0]}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(temp, label, (ix + 4, iy + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow(WINDOW_NAME, temp)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        w, h = abs(x - ix), abs(y - iy)
        if w > 20 and h > 20:
            rx, ry = min(ix, x), min(iy, y)
            label = f"Car {label_counter[0]}"
            rois.append({"label": label, "x": rx, "y": ry, "w": w, "h": h})
            label_counter[0] += 1
            print(f"  Added ROI: {label} = ({rx}, {ry}, {w}, {h})")
            display[:] = draw_rois(frame, rois)
            cv2.imshow(WINDOW_NAME, display)


def main():
    global current_label

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: Cannot open {VIDEO_PATH}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Cannot read first frame")
        return

    h, w = frame.shape[:2]
    print(f"Video frame size: {w}x{h}")
    print()
    print("=== ROI Selection Instructions ===")
    print("1. Click and drag to draw a rectangle around each car")
    print("2. Press 'u' to undo the last ROI")
    print("3. Press 'c' to clear all ROIs")
    print("4. Press 's' to save ROIs and exit")
    print("5. Press 'q' or ESC to quit without saving")
    print()

    display = draw_rois(frame, rois)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, w // 2, h // 2)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback,
                         {"frame": frame.copy(), "display": display})
    cv2.imshow(WINDOW_NAME, display)

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            print("Quit without saving.")
            break
        elif key == ord("s"):
            if not rois:
                print("No ROIs defined. Draw at least one rectangle.")
                continue
            with open(ROI_CONFIG_PATH, "w") as f:
                json.dump({"rois": rois, "frame_size": [w, h]}, f, indent=2)
            print(f"\nSaved {len(rois)} ROI(s) to {ROI_CONFIG_PATH}")
            for r in rois:
                print(f"  {r['label']}: ({r['x']}, {r['y']}, {r['w']}, {r['h']})")
            break
        elif key == ord("u"):
            if rois:
                removed = rois.pop()
                print(f"  Removed ROI: {removed['label']}")
                display[:] = draw_rois(frame, rois)
                cv2.imshow(WINDOW_NAME, display)
        elif key == ord("c"):
            rois.clear()
            label_counter[0] = 1
            print("  Cleared all ROIs")
            display[:] = draw_rois(frame, rois)
            cv2.imshow(WINDOW_NAME, display)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
