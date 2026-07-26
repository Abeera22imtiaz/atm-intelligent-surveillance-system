"""
tracker.py

Trash-only tracking for ATM surveillance using YOLOv8 detections + a custom
spatial-memory tracker (NOT a standard motion-based tracker like ByteTrack/BoT-SORT).

Run directly:
    python tracker.py
(or import run_atms_detection() from it, same as xvision_detect.py)

Reuses shared paths/thresholds from src/config.py — see the additions listed
at the top of config.py for the new settings this file expects.

Design rationale:
    Trash is a STATIC object. It does not move on its own. If it "disappears"
    from a frame it is because a person is physically blocking the camera's
    view of it, not because it moved. So instead of predicting motion (Kalman
    filter style), we track trash by REMEMBERED LOCATION, and use person
    detections purely as an "is this occlusion?" signal. This is cheap
    (no ReID / appearance model needed) and fits Google Colab Free (T4) limits.

Pipeline:
    1. Run YOLOv8 on each frame -> separate trash detections and person detections.
    2. Match trash detections to existing tracked trash objects by centroid distance.
    3. Unmatched existing trash objects:
         - if a person box overlaps/near their last known position -> mark OCCLUDED,
           keep the ID, KEEP the dwell timer running (trash assumed still present).
         - if no person is nearby -> start/continue a grace-period countdown;
           if the grace period expires -> mark REMOVED, freeze the dwell timer.
    4. Unmatched detections (no existing trash object nearby) -> new trash ID.
    5. Dwell time = (last_active_time - first_seen_time), frozen once REMOVED.
       If dwell time crosses TRASH_ALARM_THRESHOLD_SECS -> alarm.
    6. On video end, generate a text report for every trash ID ever tracked.
"""

import os
import cv2
import time
import torch
from ultralytics import YOLO

from src.config import (
    BASE_PATH,
    VIDEO_SOURCE,
    OUTPUT_VIDEO_PATH,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    TRASH_ALARM_THRESHOLD_SECS,
    CLASS_MAP,
    MODEL_WEIGHTS_PATH,
    REPORT_PATH,
    OCCLUSION_GRACE_SECS,
    MATCH_DIST_RATIO,
    PERSON_PROXIMITY_RATIO,
    REOPEN_WINDOW_SECS,
)

# ----------------------------------------------------------------------------
# Class ids, derived from config so this stays in sync with CLASS_MAP.
# ----------------------------------------------------------------------------
_NAME_TO_ID = {v: k for k, v in CLASS_MAP.items()}
TRASH_CLASS_ID = _NAME_TO_ID["trash"]
PERSON_CLASS_ID = _NAME_TO_ID["person"]



# ----------------------------------------------------------------------------
# Small geometry helpers
# ----------------------------------------------------------------------------
def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def distance(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def format_time(seconds):
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


# ----------------------------------------------------------------------------
# Trash object state
# ----------------------------------------------------------------------------
class TrashObject:
    def __init__(self, obj_id, bbox, current_time):
        self.id = obj_id
        self.bbox = bbox                     # last known bounding box
        self.status = "visible"              # visible | occluded | removed
        self.first_seen_time = current_time
        self.last_active_time = current_time  # last time we consider it "still there"
        self.missed_time_start = None        # when un-occluded absence started
        self.alarm_triggered = False
        self.alarm_time = None
        self.removed_at = None
        self.frozen_dwell = None             # set once removed

    def dwell_time(self, current_time):
        if self.status == "removed":
            return self.frozen_dwell
        return current_time - self.first_seen_time

    def mark_visible(self, bbox, current_time):
        self.bbox = bbox
        self.status = "visible"
        self.last_active_time = current_time
        self.missed_time_start = None

    def mark_occluded(self, current_time):
        # Position frozen at last known bbox; timer keeps running via last_active_time.
        self.status = "occluded"
        self.last_active_time = current_time
        self.missed_time_start = None  # reset grace clock: a person is validly explaining the absence

    def mark_removed(self, current_time):
        self.status = "removed"
        self.removed_at = current_time
        self.frozen_dwell = self.last_active_time - self.first_seen_time


# ----------------------------------------------------------------------------
# Spatial-memory tracker
# ----------------------------------------------------------------------------
class TrashSpatialTracker:
    def __init__(self, frame_w, frame_h):
        self.next_id = 1
        self.active = {}       # id -> TrashObject (visible or occluded)
        self.history = []      # removed TrashObjects
        diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
        self.match_dist = diag * MATCH_DIST_RATIO
        self.person_proximity = diag * PERSON_PROXIMITY_RATIO

    def _person_near(self, bbox, person_boxes):
        # Expand the trash box by the proximity margin on all sides, then check
        # for a plain rectangle overlap with each person box.
        #
        # We deliberately do NOT use centroid-to-centroid distance here: a
        # standing person's box centroid sits around chest/waist height,
        # while ground-level trash's centroid sits much lower. Those two
        # centroids can be far apart even while the person's body is directly
        # blocking the camera's view of the trash — which was causing real
        # occlusions to go undetected and trigger false "removed" events.
        x1, y1, x2, y2 = bbox
        ex1 = x1 - self.person_proximity
        ey1 = y1 - self.person_proximity
        ex2 = x2 + self.person_proximity
        ey2 = y2 + self.person_proximity

        for px1, py1, px2, py2 in person_boxes:
            overlap = not (px2 < ex1 or px1 > ex2 or py2 < ey1 or py1 > ey2)
            if overlap:
                return True
        return False

    def _try_revive(self, det, current_time):
        """
        If a trash object was marked 'removed' very recently, and this new
        detection lands close to where it was, treat it as the SAME object
        reappearing after a brief detection dropout rather than a new item.
        Returns the revived TrashObject, or None if nothing qualifies.
        """
        det_c = centroid(det)
        best_match = None
        best_dist = None
        for obj in self.history:
            if obj.removed_at is None:
                continue
            if current_time - obj.removed_at > REOPEN_WINDOW_SECS:
                continue
            d = distance(det_c, centroid(obj.bbox))
            if d <= self.match_dist and (best_dist is None or d < best_dist):
                best_match, best_dist = obj, d

        if best_match is not None:
            self.history.remove(best_match)
            best_match.mark_visible(det, current_time)
            best_match.status = "visible"
            best_match.removed_at = None
            best_match.frozen_dwell = None
            return best_match
        return None

    def update(self, trash_boxes, person_boxes, current_time):
        unmatched_dets = list(trash_boxes)
        matched_ids = set()

        # 1. Try to match existing active trash objects to a detection this frame.
        for obj_id, obj in self.active.items():
            if not unmatched_dets:
                break
            obj_c = centroid(obj.bbox)
            best_idx, best_dist = None, None
            for i, det in enumerate(unmatched_dets):
                d = distance(obj_c, centroid(det))
                if d <= self.match_dist and (best_dist is None or d < best_dist):
                    best_idx, best_dist = i, d
            if best_idx is not None:
                det = unmatched_dets.pop(best_idx)
                obj.mark_visible(det, current_time)
                matched_ids.add(obj_id)

        # 2. Handle active objects that were NOT matched this frame.
        to_remove = []
        for obj_id, obj in self.active.items():
            if obj_id in matched_ids:
                continue
            if self._person_near(obj.bbox, person_boxes):
                # A person is plausibly blocking it -> occlusion, keep timer running.
                obj.mark_occluded(current_time)
            else:
                # No detection, no person nearby -> start/continue grace countdown.
                if obj.missed_time_start is None:
                    obj.missed_time_start = current_time
                elif current_time - obj.missed_time_start >= OCCLUSION_GRACE_SECS:
                    obj.mark_removed(current_time)
                    to_remove.append(obj_id)

        for obj_id in to_remove:
            self.history.append(self.active.pop(obj_id))

        # 3. Any detections left over are new trash objects — UNLESS they land
        #    on a spot where a trash object was removed very recently. That's
        #    most likely the same physical trash that got missed for a few
        #    frames (confidence dip, motion blur) right at the removal
        #    boundary, not a genuinely new item. Revive the old ID in that case.
        for det in unmatched_dets:
            revived = self._try_revive(det, current_time)
            if revived is not None:
                self.active[revived.id] = revived
                continue
            new_obj = TrashObject(self.next_id, det, current_time)
            self.active[self.next_id] = new_obj
            self.next_id += 1

        # 4. Alarm check for every currently active object.
        for obj in self.active.values():
            dwell = obj.dwell_time(current_time)
            if dwell >= TRASH_ALARM_THRESHOLD_SECS and not obj.alarm_triggered:
                obj.alarm_triggered = True
                obj.alarm_time = current_time

    def finalize(self, current_time):
        # Freeze dwell time for anything still active when the video ends.
        for obj in self.active.values():
            if obj.status != "removed":
                obj.frozen_dwell = obj.dwell_time(current_time)


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def draw_trash_box(frame, obj, current_time):
    x1, y1, x2, y2 = [int(v) for v in obj.bbox]
    dwell = obj.dwell_time(current_time)

    if obj.alarm_triggered:
        color = (0, 0, 255)          # red
        status_label = "ALARM - OCCLUDED" if obj.status == "occluded" else "ALARM - TRASH"
    elif obj.status == "occluded":
        color = (0, 165, 255)        # orange
        status_label = "OCCLUDED"
    else:
        color = (0, 200, 0)          # green
        status_label = "TRACKED"

    thickness = 3 if obj.status != "occluded" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = f"Trash #{obj.id} | {format_time(dwell)} | {status_label}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_person_box(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
    cv2.putText(frame, "person", (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)


def draw_alarm_banner(frame, active_alarm_count):
    if active_alarm_count > 0:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 255), -1)
        cv2.putText(frame, f"ALARM ACTIVE - {active_alarm_count} trash object(s) over threshold",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ----------------------------------------------------------------------------
# Report generation
# ----------------------------------------------------------------------------
def generate_report(tracker, report_path):
    all_objects = list(tracker.history) + list(tracker.active.values())
    all_objects.sort(key=lambda o: o.id)

    lines = []
    for obj in all_objects:
        lines.append(f"Trash ID: {obj.id}")
        lines.append(f"First Seen: {format_time(obj.first_seen_time)}")
        if obj.status == "removed":
            lines.append(f"Removed At: {format_time(obj.removed_at)}")
            lines.append(f"Total Duration: {int(obj.frozen_dwell)} seconds")
            lines.append("Status: Removed")
        else:
            lines.append(f"Total Duration: {int(obj.frozen_dwell)} seconds")
            lines.append("Status: Still Present")
        if obj.alarm_triggered:
            lines.append(f"Alarm: Triggered at {format_time(obj.alarm_time)}")
        else:
            lines.append("Alarm: Not Triggered")
        lines.append("")  # blank line between entries

    report_text = "\n".join(lines) if lines else "No trash objects were detected in this video.\n"

    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + "=" * 50)
    print("TRASH DWELL-TIME REPORT")
    print("=" * 50)
    print(report_text)
    print(f"Report saved to: {report_path}")


# ----------------------------------------------------------------------------
# Main detection + tracking loop
# ----------------------------------------------------------------------------
def run_trash_tracking():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Trained weights not found at: {MODEL_WEIGHTS_PATH}")
    if not os.path.exists(VIDEO_SOURCE):
        raise FileNotFoundError(f"Video source not found at: {VIDEO_SOURCE}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Loading trained model... (device: {'GPU' if device == 0 else 'CPU'})")
    model = YOLO(MODEL_WEIGHTS_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_SOURCE}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_w, frame_h))

    tracker = TrashSpatialTracker(frame_w, frame_h)

    print(f"Processing video: {VIDEO_SOURCE}")
    print(f"Resolution: {frame_w}x{frame_h} | FPS: {fps:.2f} | Frames: {total_frames}")

    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps  # video-time seconds, not wall-clock

        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=device,
            verbose=False,
        )[0]

        trash_boxes, person_boxes = [], []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            if cls_id == TRASH_CLASS_ID:
                trash_boxes.append(xyxy)
            elif cls_id == PERSON_CLASS_ID:
                person_boxes.append(xyxy)

        tracker.update(trash_boxes, person_boxes, current_time)

        for pbox in person_boxes:
            draw_person_box(frame, pbox)

        active_alarms = 0
        for obj in tracker.active.values():
            draw_trash_box(frame, obj, current_time)
            if obj.alarm_triggered:
                active_alarms += 1
        draw_alarm_banner(frame, active_alarms)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  Frame {frame_idx}/{total_frames} | video-time {format_time(current_time)} "
                  f"| active trash: {len(tracker.active)} | elapsed {elapsed:.1f}s")

    final_time = frame_idx / fps
    tracker.finalize(final_time)

    cap.release()
    writer.release()

    print("\n" + "=" * 50)
    print(f"PROCESSING COMPLETE. Output video saved to: {OUTPUT_VIDEO_PATH}")
    print("=" * 50)

    generate_report(tracker, REPORT_PATH)


if __name__ == "__main__":
    run_trash_tracking()
