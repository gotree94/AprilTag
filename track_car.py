import cv2
import numpy as np
import pupil_apriltags as apriltag
import csv
import json
import time
from pathlib import Path

VIDEO_PATH = Path(__file__).parent / "cam.mp4"
OUTPUT_VIDEO_PATH = Path(__file__).parent / "output.mp4"
OUTPUT_CSV_PATH = Path(__file__).parent / "position.csv"
ROI_CONFIG_PATH = Path(__file__).parent / "roi_config.json"

TAG_IDS = [0, 1, 2, 3]

# Known physical layout of the 4 tags (in mm).
# Arrange IDs as: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
# Adjust these values to match your actual tag placement.
WORLD_PTS_MM = np.float32([
    [0,         0],          # ID 0 — top-left
    [1030,      0],          # ID 1 — top-right
    [1030,      1030],       # ID 2 — bottom-right
    [0,         1030],       # ID 3 — bottom-left
])

# Temporal smoothing: remember last known tag positions
_last_tag_positions = {}
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _enhance(gray):
    return _clahe.apply(gray)


def detect_tags(gray, detector):
    global _last_tag_positions

    merged = {}
    for r in detector.detect(gray):
        if r.tag_id in TAG_IDS:
            merged[r.tag_id] = r

    missing = [tid for tid in TAG_IDS if tid not in merged]
    if missing:
        for r in detector.detect(_enhance(gray)):
            if r.tag_id in missing:
                merged[r.tag_id] = r
                missing.remove(r.tag_id)

    for tid in missing[:]:
        if tid in _last_tag_positions:
            lx, ly = _last_tag_positions[tid]
            x1 = max(0, int(lx - 150))
            y1 = max(0, int(ly - 150))
            x2 = min(gray.shape[1], int(lx + 150))
            y2 = min(gray.shape[0], int(ly + 150))
            if x2 > x1 and y2 > y1:
                for roi in (gray[y1:y2, x1:x2], _enhance(gray[y1:y2, x1:x2])):
                    for r in detector.detect(roi):
                        if r.tag_id == tid:
                            r.center[0] += x1
                            r.center[1] += y1
                            if r.corners is not None:
                                r.corners[:, 0] += x1
                                r.corners[:, 1] += y1
                            merged[tid] = r
                            missing.remove(tid)
                            break
                    if tid in merged:
                        break

    _last_tag_positions = {
        tid: (r.center[0], r.center[1]) for tid, r in merged.items()
    }

    return merged


def compute_car_center(tags):
    if not tags:
        return None
    centers = np.float32([r.center for r in tags.values()])
    return centers.mean(axis=0)


def calibrate_homography(tags):
    present = [tid for tid in TAG_IDS if tid in tags]
    if len(present) < 4:
        return None
    img_pts = np.float32([tags[tid].center for tid in present])
    world_pts = np.float32([WORLD_PTS_MM[tid] for tid in present])
    H, _ = cv2.findHomography(img_pts, world_pts, method=0)
    return H


def estimate_floor_position(tags, H):
    if tags is None or H is None:
        return None
    car_img = compute_car_center(tags)
    if car_img is None:
        return None
    car_world = cv2.perspectiveTransform(
        car_img.reshape(1, 1, 2).astype(np.float32), H
    ).reshape(2)
    return car_world

TAG_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]

# Car tracker state
_car_tracker = {}
_CAR_THRESH = 80
_CAR_DEFAULT_MIN_AREA = 400

# Load ROIs from config file or fall back to hardcoded defaults
_FALLBACK_ROIS = [
    {"label": "Car 1", "x": 550, "y": 200, "w": 200, "h": 250},
    {"label": "Car 2", "x": 950, "y": 888, "w": 130, "h": 22},
]


def _roi_color(index):
    palette = [(0, 200, 255), (255, 200, 0), (0, 255, 100),
               (255, 100, 255), (100, 200, 255)]
    return palette[index % len(palette)]


def load_roi_config():
    try:
        with open(ROI_CONFIG_PATH) as f:
            data = json.load(f)
        rois = data.get("rois", [])
        if rois:
            print(f"Loaded {len(rois)} ROI(s) from {ROI_CONFIG_PATH}")
            for r in rois:
                print(f"  {r['label']}: ({r['x']}, {r['y']}, {r['w']}, {r['h']})")
            return rois
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    print("Using default ROI configuration")
    return _FALLBACK_ROIS.copy()


def detect_cars(gray, rois):
    global _car_tracker

    _, dark = cv2.threshold(gray, _CAR_THRESH, 255, cv2.THRESH_BINARY_INV)

    results = {}

    for r in rois:
        label = r["label"]
        rx, ry, rw, rh = r["x"], r["y"], r["w"], r["h"]
        roi = dark[ry:ry+rh, rx:rx+rw]
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = r.get("min_area", _CAR_DEFAULT_MIN_AREA)
        best_cnt = None
        best_a = 0
        best_box = None
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            gx, gy = x + rx, y + ry
            if best_cnt is None or a > best_a:
                best_cnt = c
                best_a = a
                best_box = (gx, gy, w, h)
        if best_cnt is not None:
            cnt, a, (gx, gy, w, h) = best_cnt, best_a, best_box
            angle = 0.0
            if len(cnt) >= 5:
                rect = cv2.minAreaRect(cnt)
                (rw, rh), ra = rect[1], rect[2]
                if rw < rh:
                    ra += 90
                angle = ra % 180
                prev = _car_tracker.get(label, {}).get("angle")
                if prev is not None and abs(angle - prev) > 90:
                    angle = angle - 180 if angle > prev else angle + 180
            cx, cy = gx + w // 2, gy + h // 2
            results[label] = {
                "bbox": (gx, gy, w, h),
                "centroid": (cx, cy),
                "area": a,
                "angle": angle,
            }

    _car_tracker = results
    return results


def draw_overlay(frame, tags, car_center_img, car_center_world, H, fps, cars=None):
    h, w = frame.shape[:2]

    car_labels = list(cars.keys()) if cars else []
    car_angles = {}
    if cars:
        for i, (label, info) in enumerate(cars.items()):
            gx, gy, bw, bh = info["bbox"]
            cx, cy = info["centroid"]
            angle = info.get("angle", 0)
            car_angles[label] = angle
            color = _roi_color(i)

            cv2.rectangle(frame, (gx, gy), (gx + bw, gy + bh), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)

            coord_text = f"({cx},{cy})"
            (tw, th), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(frame, coord_text, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            lx, ly = gx, max(gy - 8, 20)
            cv2.rectangle(frame, (lx - 2, ly - lh - 2), (lx + lw + 2, ly + 2), color, -1)
            cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            a_disp = angle % 360
            a_text = f"{a_disp:.0f}deg"
            (aw, ah), _ = cv2.getTextSize(a_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            ax2 = min(gx + bw - aw - 4, w - 10)
            ay2 = gy + bh - 6
            cv2.rectangle(frame, (ax2 - 2, ay2 - ah - 2), (ax2 + aw + 2, ay2 + 2), color, -1)
            cv2.putText(frame, a_text, (ax2, ay2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            rad = np.deg2rad(angle)
            arrow_len = max(bw, bh) // 2 + 10
            ax = int(cx + arrow_len * np.cos(rad))
            ay = int(cy + arrow_len * np.sin(rad))
            cv2.arrowedLine(frame, (cx, cy), (ax, ay), color, 2, cv2.LINE_AA, tipLength=0.3)

            dist_lines = []
            for tid, r in tags.items():
                if H is not None:
                    car_w = cv2.perspectiveTransform(
                        np.float32([[[cx, cy]]]), H).reshape(2)
                    tag_w = cv2.perspectiveTransform(
                        r.center.reshape(1, 1, 2).astype(np.float32), H).reshape(2)
                    dmm = np.linalg.norm(car_w - tag_w)
                    dist_lines.append(f"T{tid}:{dmm:.0f}mm")
                else:
                    dpx = np.linalg.norm(np.float32([cx - r.center[0], cy - r.center[1]]))
                    dist_lines.append(f"T{tid}:{dpx:.0f}px")

            dist_text = "  ".join(dist_lines)
            (dw, dh), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            dx = min(cx - dw // 2, w - dw - 10)
            dy = min(gy + bh + dh + 14, h - 10)
            dx = max(dx, 10)
            cv2.rectangle(frame, (dx - 2, dy - dh - 2), (dx + dw + 2, dy + 2), (0, 0, 0), -1)
            cv2.putText(frame, dist_text, (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if len(car_labels) >= 2 and "Car 1" in cars and "Car 2" in cars:
        c1 = cars["Car 1"]
        c2 = cars["Car 2"]
        x1, y1 = c1["centroid"]
        x2, y2 = c2["centroid"]
        a1 = car_angles.get("Car 1", 0)
        dx = x2 - x1
        dy = y2 - y1
        dist_px = np.linalg.norm(np.float32([dx, dy]))
        target_deg = np.rad2deg(np.arctan2(dy, dx))
        steer = (target_deg - a1 + 180) % 360 - 180

        if H is not None:
            w1 = cv2.perspectiveTransform(np.float32([[[x1, y1]]]), H).reshape(2)
            w2 = cv2.perspectiveTransform(np.float32([[[x2, y2]]]), H).reshape(2)
            dist_mm = np.linalg.norm(w2 - w1)
            dist_str = f"{dist_mm:.0f}mm"
        else:
            dist_str = f"{dist_px:.0f}px"

        cv2.arrowedLine(frame, (x1, y1), (x2, y2), (100, 255, 255), 2, cv2.LINE_AA, tipLength=0.06)
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        heading_disp = target_deg % 360
        nav_text = f"Target: {dist_str}  Heading: {heading_disp:.0f}deg  Steer: {steer:+.0f}deg"
        (nw, nh), _ = cv2.getTextSize(nav_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        nx = min(max(mid_x - nw // 2, 10), w - nw - 10)
        ny = min(mid_y - 10, h - 10)
        cv2.rectangle(frame, (nx - 2, ny - nh - 2), (nx + nw + 2, ny + 2), (100, 100, 0), -1)
        cv2.putText(frame, nav_text, (nx, ny), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

    for tid, r in tags.items():
        corners = r.corners.astype(int)
        color = (0, 255, 0)
        for i in range(4):
            cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color, 2)
        cx, cy = int(r.center[0]), int(r.center[1])
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
        cv2.putText(frame, str(tid), (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if car_center_img is not None:
            cv2.arrowedLine(frame, (cx, cy), tuple(car_center_img.astype(int)),
                            (255, 100, 100), 2, cv2.LINE_AA, tipLength=0.08)

            dpx = car_center_img[0] - cx
            dpy = car_center_img[1] - cy

            if H is not None:
                tag_world = cv2.perspectiveTransform(
                    r.center.reshape(1, 1, 2).astype(np.float32), H
                ).reshape(2)
                dmx = car_center_world[0] - tag_world[0]
                dmy = car_center_world[1] - tag_world[1]
                rel = f"dx={dmx:+.0f} dy={dmy:+.0f}mm"
            else:
                rel = f"dx={dpx:+.0f} dy={dpy:+.0f}px"

            lx = int(cx + dpx * 0.35)
            ly = int(cy + dpy * 0.35)
            (tw, th), _ = cv2.getTextSize(rel, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            bx, by = lx - tw // 2, ly - th // 2
            cv2.rectangle(frame, (bx - 2, by - 2), (bx + tw + 2, by + th + 2),
                          (0, 0, 0), -1)
            cv2.putText(frame, rel, (bx, by + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 150), 1)

    for i, j in TAG_EDGES:
        if i in tags and j in tags:
            p1 = tags[i].center
            p2 = tags[j].center
            dist_px = np.linalg.norm(p1 - p2)
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)),
                     (255, 200, 0), 2, cv2.LINE_AA)
            label = f"103cm  ({dist_px:.0f}px)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            bx, by = int(mid[0] - tw / 2), int(mid[1] - th / 2)
            cv2.rectangle(frame, (bx - 2, by - 2), (bx + tw + 2, by + th + 2),
                          (0, 0, 0), -1)
            cv2.putText(frame, label, (bx, by + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    if car_center_img is not None:
        cx, cy = int(car_center_img[0]), int(car_center_img[1])
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        cv2.circle(frame, (cx, cy), 12, (0, 0, 255), 2)

    if car_center_world is not None:
        info = f"Pos: ({car_center_world[0]:.0f}, {car_center_world[1]:.0f}) mm"
    else:
        info = "Pos: N/A"
    if car_center_img is not None:
        info += f" | Pixel: ({int(car_center_img[0])}, {int(car_center_img[1])})"
    info += f" | FPS: {fps:.1f}"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    frame[:50] = cv2.addWeighted(frame[:50], 0.3, overlay[:50], 0.7, 0)
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    import sys
    headless = "--headless" in sys.argv
    if not headless:
        try:
            cv2.imshow("", np.zeros((10, 10), dtype=np.uint8))
            cv2.destroyAllWindows()
        except cv2.error:
            print("GUI not available, running headless mode.")
            headless = True

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: Cannot open {VIDEO_PATH}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {W}x{FRAME_H}, {FPS:.1f} fps, {total} frames")

    detector = apriltag.Detector(
        families="tag36h11",
        quad_decimate=1.0,
        quad_sigma=0.4,
        refine_edges=1,
        decode_sharpening=0.5,
    )

    rois = load_roi_config()

    # Find a frame with all 4 tags to calibrate the homography
    print("Calibrating homography from first frame with all 4 tags...")
    H = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detect_tags(gray, detector)
        H = calibrate_homography(tags)
        if H is not None:
            calib_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            print(f"  Calibrated from frame {calib_frame}")
            break
    if H is None:
        print("  Could not calibrate (need all 4 tags visible). Using pixel coordinates only.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, FRAME_H))

    csv_fields = ["frame",
                   "tag0_x", "tag0_y", "tag1_x", "tag1_y",
                   "tag2_x", "tag2_y", "tag3_x", "tag3_y",
                   "car_x_mm", "car_y_mm", "car_x_px", "car_y_px"]
    for r in rois:
        label = r["label"].replace(" ", "_")
        csv_fields += [f"{label}_x", f"{label}_y", f"{label}_w", f"{label}_h",
                       f"{label}_angle"]

    csv_fp = open(OUTPUT_CSV_PATH, "w", newline="")
    csv_w = csv.writer(csv_fp)
    csv_w.writerow(csv_fields)

    frame_idx = 0
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detect_tags(gray, detector)
        cars = detect_cars(gray, rois)
        car_center_img = compute_car_center(tags)
        car_center_world = estimate_floor_position(tags, H)

        row = [frame_idx]
        row += [(tags[t].center[0] if t in tags else "") for t in TAG_IDS]
        row += [(tags[t].center[1] if t in tags else "") for t in TAG_IDS]
        row += [f"{car_center_world[0]:.1f}" if car_center_world is not None else ""]
        row += [f"{car_center_world[1]:.1f}" if car_center_world is not None else ""]
        row += [f"{car_center_img[0]:.1f}" if car_center_img is not None else ""]
        row += [f"{car_center_img[1]:.1f}" if car_center_img is not None else ""]
        for r in rois:
            label = r["label"]
            if label in cars:
                gx, gy, bw, bh = cars[label]["bbox"]
                angle = cars[label].get("angle", "")
                row += [gx, gy, bw, bh, f"{angle:.1f}" if angle != "" else ""]
            else:
                row += ["", "", "", "", ""]
        csv_w.writerow(row)

        draw_overlay(frame, tags, car_center_img, car_center_world, H, FPS, cars)
        writer.write(frame)

        if not headless:
            cv2.imshow("Car Position Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.perf_counter() - t0
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Processed {frame_idx}/{total} frames ({fps_actual:.1f} fps)")

    cap.release()
    writer.release()
    csv_fp.close()
    if not headless:
        cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t0
    print(f"\nDone. {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"Output video: {OUTPUT_VIDEO_PATH}")
    print(f"Position data: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
