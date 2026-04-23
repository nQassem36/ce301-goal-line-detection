import cv2
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

USE_TRACKER = True

TRACKER_INIT_FRAME = 70
VIDEO_PATH = Path("video/IMG_6763 - 70.mp4")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "goal_line_ball_demo.mp4"

CANNY_LOW = 50
CANNY_HIGH = 150

HOUGH_THRESHOLD = 120
HOUGH_MIN_LINE_LEN = 400
HOUGH_MAX_LINE_GAP = 25

CIRCLE_DP = 1.2
CIRCLE_MIN_DIST = 50
CIRCLE_PARAM1 = 100
CIRCLE_PARAM2 = 30
CIRCLE_MIN_R = 8
CIRCLE_MAX_R = 70

LINE_COLOR = "white" # "yellow" or "white" depending on the video

YELLOW_LOW = (15, 80, 80)
YELLOW_HIGH = (40, 255, 255)

WHITE_LOW = (0, 0, 170)
WHITE_HIGH = (180, 70, 255)


CONFIRM_FRAMES = 2

MAX_MOVE_PER_FRAME = 300
MAX_RADIUS_CHANGE = 30

MAX_OCCLUSION_FRAMES = 20
POST_OCCLUSION_MARGIN = 110

REACQUIRE_SEARCH_MARGIN_X = 320
REACQUIRE_SEARCH_MARGIN_Y = 220
TEMPLATE_MATCH_THRESHOLD = 0.30

GOAL_SIDE = "right" # "right" or "left" depending on the video

TARGET_FRAME = 84

@dataclass
class LineState:
    x1: int
    y1: int
    x2: int
    y2: int

def _line_length(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.hypot(x2 - x1, y2 - y1)

def _line_angle_deg(x1: int, y1: int, x2: int, y2: int) -> float:
    return abs(math.degrees(math.atan2((y2 - y1), (x2 - x1))))

def detect_goal_line(frame, prev: Optional[LineState]) -> Optional[LineState]:
    frame_h, frame_w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if LINE_COLOR == "yellow":
        mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    else:
        mask = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)

    if LINE_COLOR == "yellow":
        roi_y0 = int(frame_h * 0.35)
        roi_y1 = int(frame_h * 0.98)
        roi_x0 = int(frame_w * 0.25)
        roi_x1 = int(frame_w * 0.75)
    else:
        roi_y0 = int(frame_h * 0.45)
        roi_y1 = int(frame_h * 0.98)
        roi_x0 = int(frame_w * 0.45)
        roi_x1 = int(frame_w * 0.75)

    roi = mask[roi_y0:roi_y1, roi_x0:roi_x1]

    kernel = np.ones((5, 5), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)

    best_idx = -1
    best_score = -1

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 80:
            continue
        if h_box < int((roi_y1 - roi_y0) * 0.35):
            continue

        score = h_box * 3 + area

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx == -1:
        return prev

    component_mask = (labels == best_idx).astype(np.uint8) * 255
    ys, xs = np.where(component_mask > 0)

    xs_full = xs + roi_x0
    ys_full = ys + roi_y0

    if len(xs_full) < 20:
        return prev

    fit = np.polyfit(ys_full, xs_full, 1)
    m, c = fit

    y_top = int(np.min(ys_full))
    y_bottom = int(np.max(ys_full))

    x_top = int(m * y_top + c)
    x_bottom = int(m * y_bottom + c)

    if prev is not None:
        x_top = int(0.8 * prev.x1 + 0.2 * x_top)
        y_top = int(0.8 * prev.y1 + 0.2 * y_top)
        x_bottom = int(0.8 * prev.x2 + 0.2 * x_bottom)
        y_bottom = int(0.8 * prev.y2 + 0.2 * y_bottom)

    return LineState(x_top, y_top, x_bottom, y_bottom)

def detect_ball(gray, edges, prev_center: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int, int]]:
    h, w = gray.shape

    if prev_center is not None:
        px, py = prev_center
        margin_x = int(w * 0.25)
        margin_y = int(h * 0.25)
        x0 = max(0, px - margin_x)
        x1 = min(w, px + margin_x)
        y0 = max(0, py - margin_y)
        y1 = min(h, py + margin_y)
        roi = gray[y0:y1, x0:x1]
        offx, offy = x0, y0
    else:
        roi = gray
        offx, offy = 0, 0

    blur = cv2.GaussianBlur(roi, (9, 9), 2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=CIRCLE_DP,
        minDist=CIRCLE_MIN_DIST,
        param1=CIRCLE_PARAM1,
        param2=CIRCLE_PARAM2,
        minRadius=CIRCLE_MIN_R,
        maxRadius=CIRCLE_MAX_R,
    )

    if circles is None:
        return None

    circles = circles[0]

    best = None
    best_score = -1e9

    for (x, y, r) in circles:
        cx = int(x + offx)
        cy = int(y + offy)
        rr = int(r)

        if cy < int(h * 0.55):
            continue

        score = ball_edge_score(edges, cx, cy, rr)

        if score > best_score:
            best_score = score
            best = (cx, cy, rr)

    if best is None:
        return None

    return best

def ball_edge_score(edges, bx: int, by: int, br: int) -> float:
    x0 = max(0, bx - br)
    x1 = min(edges.shape[1], bx + br)
    y0 = max(0, by - br)
    y1 = min(edges.shape[0], by + br)
    patch = edges[y0:y1, x0:x1]
    return float(patch.mean())

def whole_ball_crossed(line: LineState, ball: Tuple[int, int, int]) -> bool:
    CROSSING_TOLERANCE = 6

    bx, by, br = ball

    if (line.y2 - line.y1) == 0:
        return False

    t = (by - line.y1) / (line.y2 - line.y1)
    x_line_at_ball = line.x1 + t * (line.x2 - line.x1)

    if GOAL_SIDE == "right":
        return (bx - br) > (x_line_at_ball - CROSSING_TOLERANCE)
    else:
        return (bx + br) < (x_line_at_ball + CROSSING_TOLERANCE)
    
def update_ball_template(frame, ball, padding=8):
    if ball is None:
        return None

    bx, by, br = ball
    h, w = frame.shape[:2]

    x0 = max(0, bx - br - padding)
    y0 = max(0, by - br - padding)
    x1 = min(w, bx + br + padding)
    y1 = min(h, by + br + padding)

    if x1 <= x0 or y1 <= y0:
        return None

    patch = frame[y0:y1, x0:x1].copy()

    if patch.size == 0:
        return None

    return patch

def predict_ball_from_history(ball_history):
    if len(ball_history) < 2:
        return None

    x1, y1, r1 = ball_history[-2]
    x2, y2, r2 = ball_history[-1]

    dx = x2 - x1
    dy = y2 - y1
    dr = r2 - r1

    pred_x = x2 + dx
    pred_y = y2 + dy
    pred_r = max(5, r2 + dr)

    return (int(pred_x), int(pred_y), int(pred_r))

def reacquire_ball_with_template(frame, template, last_ball):
    if template is None or last_ball is None:
        return None

    h, w = frame.shape[:2]
    th, tw = template.shape[:2]

    lbx, lby, lbr = last_ball

    x0 = max(0, lbx - REACQUIRE_SEARCH_MARGIN_X)
    x1 = min(w, lbx + REACQUIRE_SEARCH_MARGIN_X)
    y0 = max(0, lby - REACQUIRE_SEARCH_MARGIN_Y)
    y1 = min(h, lby + REACQUIRE_SEARCH_MARGIN_Y)

    search = frame[y0:y1, x0:x1]

    if search.shape[0] < th or search.shape[1] < tw:
        return None

    search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < TEMPLATE_MATCH_THRESHOLD:
        return None

    rx, ry = max_loc
    x = x0 + rx
    y = y0 + ry
    ww = tw
    hh = th

    bx = x + ww // 2
    by = y + hh // 2
    br = max(5, min(ww, hh) // 2)

    return (bx, by, br, (x, y, ww, hh))

def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
        isColor=True,
    )

    prev_line: Optional[LineState] = None
    prev_ball_center: Optional[Tuple[int, int]] = None

    consecutive_goal_frames = 0
    goal_confirmed = False
    goal_frame_index = None

    tracker = None
    tracker_active = False
    tracked_bbox = None
    tracker_warmup = 0

    occlusion_frames = 0
    last_ball = None
    ball_template = None
    ball_history = []
    MAX_HISTORY = 5

    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

        line = detect_goal_line(frame, prev=prev_line)
        ball = None

        if USE_TRACKER and (not tracker_active) and (frame_i == TRACKER_INIT_FRAME):

            cv2.imshow("Goal Line + Ball Demo", frame)
            cv2.waitKey(1)

            while True:
                bbox = cv2.selectROI(
                    "Draw a square around the ball then press ENTER",
                    frame,
                    fromCenter=False,
                    showCrosshair=True
                )

                x, y, ww, hh = [int(v) for v in bbox]

                if ww <= 0 or hh <= 0:
                    print("[WARN] Empty selection.")
                    continue

                try:
                    tracker = cv2.TrackerCSRT_create()
                except AttributeError:
                    raise RuntimeError("CSRT tracker not available. Install opencv-contrib-python.")

                tracker.init(frame, (x, y, ww, hh))
                tracker_active = True
                tracked_bbox = (x, y, ww, hh)

                bx0 = x + ww // 2
                by0 = y + hh // 2
                br0 = max(5, min(ww, hh) // 2)

                prev_ball_center = (bx0, by0)
                tracker_warmup = 3

                ball_template = frame[y:y + hh, x:x + ww].copy()
                last_ball = (bx0, by0, br0)

                ball_history.append((bx0, by0, br0))
                if len(ball_history) > MAX_HISTORY:
                    ball_history.pop(0)

                print(f"[INFO] Tracker initialised at frame {frame_i}")
                break

            cv2.destroyWindow("Draw a square around the ball then press ENTER")
            cv2.imshow("Goal Line + Ball Demo", frame)
            cv2.waitKey(1)

            frame_i += 1
            continue

        if USE_TRACKER and tracker_active:
            ok_t, bbox = tracker.update(frame)

            if ok_t:
                x, y, ww, hh = [int(v) for v in bbox]

                bx = x + ww // 2
                by = y + hh // 2
                br = max(5, min(ww, hh) // 2)

                if tracker_warmup > 0:
                    tracker_warmup -= 1
                    tracked_bbox = (x, y, ww, hh)
                    ball = (bx, by, br)
                    prev_ball_center = (bx, by)
                    last_ball = (bx, by, br)
                    occlusion_frames = 0
                else:
                    valid = True

                    aspect_ratio = ww / max(hh, 1)
                    if aspect_ratio < 0.5 or aspect_ratio > 1.8:
                        valid = False

                    if prev_ball_center is not None:
                        px, py = prev_ball_center
                        move_dist = math.hypot(bx - px, by - py)
                        if move_dist > MAX_MOVE_PER_FRAME:
                            valid = False

                    if valid:
                        tracked_bbox = (x, y, ww, hh)
                        ball = (bx, by, br)
                        prev_ball_center = (bx, by)
                        last_ball = (bx, by, br)
                        occlusion_frames = 0
                    else:
                        x_line = int((line.x1 + line.x2) / 2) if line is not None else None

                        if last_ball is not None and x_line is not None:
                            lbx, lby, lbr = last_ball
                            near_post_zone = abs(lbx - x_line) < POST_OCCLUSION_MARGIN

                            if near_post_zone and occlusion_frames < MAX_OCCLUSION_FRAMES:
                                occlusion_frames += 1

                                reacquired = reacquire_ball_with_template(frame, ball_template, last_ball)

                                if reacquired is not None:
                                    bx, by, br, new_bbox = reacquired
                                    ball = (bx, by, br)
                                    tracked_bbox = new_bbox
                                    prev_ball_center = (bx, by)
                                    last_ball = (bx, by, br)

                                    try:
                                        tracker = cv2.TrackerCSRT_create()
                                    except AttributeError:
                                        raise RuntimeError("CSRT tracker not available. Install opencv-contrib-python.")

                                    tracker.init(frame, new_bbox)
                                    tracker_active = True
                                    tracker_warmup = 2
                                    occlusion_frames = 0
                                else:
                                    predicted = predict_ball_from_history(ball_history)
                                    if predicted is not None:
                                        ball = predicted
                                        prev_ball_center = (predicted[0], predicted[1])
                                        last_ball = predicted
                                        tracked_bbox = None
                                    else:
                                        ball = last_ball
                            else:
                                tracker_active = False
                                tracker = None
                                tracked_bbox = None
                                ball = None
                        else:
                            tracker_active = False
                            tracker = None
                            tracked_bbox = None
                            ball = None

            else:
                x_line = int((line.x1 + line.x2) / 2) if line is not None else None

                if last_ball is not None and x_line is not None:
                    lbx, lby, lbr = last_ball
                    near_post_zone = abs(lbx - x_line) < POST_OCCLUSION_MARGIN

                    if near_post_zone and occlusion_frames < MAX_OCCLUSION_FRAMES:
                        occlusion_frames += 1

                        reacquired = reacquire_ball_with_template(frame, ball_template, last_ball)

                        if reacquired is not None:
                            bx, by, br, new_bbox = reacquired
                            ball = (bx, by, br)
                            tracked_bbox = new_bbox
                            prev_ball_center = (bx, by)
                            last_ball = (bx, by, br)

                            try:
                                tracker = cv2.TrackerCSRT_create()
                            except AttributeError:
                                raise RuntimeError("CSRT tracker not available. Install opencv-contrib-python.")

                            tracker.init(frame, new_bbox)
                            tracker_active = True
                            tracker_warmup = 2
                            occlusion_frames = 0
                        else:
                            predicted = predict_ball_from_history(ball_history)
                            if predicted is not None:
                                ball = predicted
                                prev_ball_center = (predicted[0], predicted[1])
                                last_ball = predicted
                                tracked_bbox = None
                            else:
                                ball = last_ball
                    else:
                        tracker_active = False
                        tracker = None
                        tracked_bbox = None
                        ball = None
                else:
                    tracker_active = False
                    tracker = None
                    tracked_bbox = None
                    ball = None

        overlay = frame.copy()

        if line is not None:
            cv2.line(overlay, (line.x1, line.y1), (line.x2, line.y2), (0, 0, 255), 6)
            prev_line = line

        if ball is not None:
            bx, by, br = ball
            prev_ball_center = (bx, by)
            ball_history.append((bx, by, br))

            if len(ball_history) > MAX_HISTORY:
                ball_history.pop(0)
            
            cv2.circle(overlay, (bx, by), br, (0, 255, 0), 3)
            cv2.circle(overlay, (bx, by), 2, (0, 255, 0), 3)
            new_template = update_ball_template(frame, ball)
            if new_template is not None:
                ball_template = new_template

        if USE_TRACKER and tracked_bbox is not None:
            x, y, ww, hh = tracked_bbox
            cv2.rectangle(overlay, (x, y), (x + ww, y + hh), (0, 255, 255), 2)

        if (line is not None) and (ball is not None):
            if whole_ball_crossed(line, ball):
                consecutive_goal_frames += 1
            else:
                consecutive_goal_frames = 0
        else:
            consecutive_goal_frames = 0

        if (not goal_confirmed) and consecutive_goal_frames >= CONFIRM_FRAMES:
            goal_confirmed = True
            goal_frame_index = frame_i

        cv2.putText(overlay, f"Frame: {frame_i}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        status = "GOAL CONFIRMED" if goal_confirmed else f"Goal frames: {consecutive_goal_frames}/{CONFIRM_FRAMES}"
        cv2.putText(overlay, status, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if goal_confirmed else (255, 255, 255),
                    2, cv2.LINE_AA)

        if line is not None and ball is not None:
            x_line_dbg = int((line.x1 + line.x2) / 2)
            bx_dbg, by_dbg, br_dbg = ball
            cv2.putText(
                overlay,
                f"x_line={x_line_dbg}  bx={bx_dbg}  br={br_dbg}  bx-r={bx_dbg - br_dbg}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        writer.write(overlay)
        cv2.imshow("Goal Line + Ball Demo", overlay)
        
        if frame_i == TARGET_FRAME:
            save_path = OUTPUT_DIR / f"goal_line_ball_demo_frame_{TARGET_FRAME:03d}.png"
            cv2.imwrite(str(save_path), overlay)
            print(f"[INFO] Saved frame {TARGET_FRAME} to {save_path}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_i += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Saved demo output video: {OUTPUT_VIDEO}")

    if goal_confirmed:
        print("Goal")
        print(f"[INFO] Goal confirmed at frame {goal_frame_index}")
    else:
        print("No goal")

if __name__ == "__main__":
    main()