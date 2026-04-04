#!/usr/bin/env python3

"""
required packages:

pipreqs hailo_apps/python/custom_standalone_apps --print
cython_bbox==0.1.5
hailort==4.23.0
lap==0.5.13
numpy==2.4.4
opencv_python_headless==4.10.0.84
picamera2==0.3.34
scipy==1.17.1

pip show
Name: hailort
Version: 4.23.0
Summary: HailoRT
Home-page: https://hailo.ai/
Author: Hailo team
Author-email: contact@hailo.ai
License: 
Location: /usr/lib/python3/dist-packages
Requires: argcomplete, contextlib2, future, netaddr, netifaces, numpy
Required-by: 

Name: picamera2
Version: 0.3.34
Summary: The libcamera-based Python interface to Raspberry Pi cameras, based on the original Picamera library
Home-page: https://github.com/RaspberryPi/picamera2
Author: Raspberry Pi & Raspberry Pi Foundation
Author-email: picamera2@raspberrypi.com
License: BSD 2-Clause License
Location: /usr/lib/python3/dist-packages
Requires: av, jsonschema, libarchive-c, numpy, OpenEXR, PiDNG, piexif, pillow, python-prctl, simplejpeg, tqdm, videodev2
Required-by:

Name: cython_bbox
Version: 0.1.5
Summary: Standalone cython_bbox
Home-page: https://github.com/samson-wang/cython_bbox.git
Author: Samson Wang
Author-email: samson.c.wang@gmail.com
License: 
Location: /home/raspberry01/hailo-apps/venv_hailo_apps/lib/python3.13/site-packages
Requires: Cython, numpy
Required-by: hailo-apps

Name: lap
Version: 0.5.13
Summary: Linear Assignment Problem solver (LAPJV/LAPMOD).
Home-page: https://github.com/gatagat/lap
Author: gatagat, rathaROG, and co.
Author-email: 
License: BSD-2-Clause
Location: /home/raspberry01/hailo-apps/venv_hailo_apps/lib/python3.13/site-packages
Requires: numpy
Required-by: hailo-apps

Name: numpy
Version: 1.26.4
Summary: Fundamental package for array computing in Python
Home-page: https://numpy.org
Author: Travis E. Oliphant et al.
Location: /home/raspberry01/hailo-apps/venv_hailo_apps/lib/python3.13/site-packages
Requires: 
Required-by: cython_bbox, hailo-apps, hailort, lancedb, lap, opencv, opencv-python, opencv-python-headless, picamera2, pidng, scipy, simplejpeg, types-JACK-Client, types-networkx, types-seaborn, types-shapely, types-tensorflow

Name: opencv-python-headless
Version: 4.10.0.84
Summary: Wrapper package for OpenCV python bindings.
Home-page: https://github.com/opencv/opencv-python
Author: 
Author-email: 
License: Apache 2.0
Location: /home/raspberry01/hailo-apps/venv_hailo_apps/lib/python3.13/site-packages
Requires: numpy
Required-by:

Name: scipy
Version: 1.17.1
Summary: Fundamental algorithms for scientific computing in Python
Home-page: https://scipy.org/
Author: 
Author-email:
Location: /home/raspberry01/hailo-apps/venv_hailo_apps/lib/python3.13/site-packages
Requires: numpy
Required-by: hailo-apps
"""

"""
example usage:

cd hailo-apps/
source setup_env.sh
cd hailo_apps/python/standalone_apps/object_detection
./object_detection_mono.py -i usb -cr scaledsd --show-fps --hef /usr/local/hailo/resources/models/hailo8/yolov8m.hef --save-output --output-dir /dev/shm
"""

"""
object_detection_mono.py — Single-file Hailo object detection.

All logic from the original multi-file project is inlined here.
No hailo_apps package imports are needed at runtime.

Binary dependencies (install once):
    pip install hailo_platform opencv-python-headless numpy scipy lap cython_bbox

Usage examples:
    # Static image (headless implied — no display needed):
    ./object_detection_mono.py --hef /usr/local/hailo/resources/models/hailo8/yolov8m.hef -i image.png

    # USB camera, headless (Ctrl+C to stop):
    ./object_detection_mono.py --hef yolov8m.hef -i usb --headless

    # USB camera, with display window ('q' or Ctrl+C to stop):
    ./object_detection_mono.py --hef yolov8m.hef -i usb

    # USB camera, scaledsd: captures at native 960x600 then center-crops to 640x480:
    ./object_detection_mono.py --hef yolov8m.hef -i usb --camera-resolution scaledsd --headless

    # Save output video while running:
    ./object_detection_mono.py --hef yolov8m.hef -i usb --headless --save-output

    # Enable tracking + FPS counter:
    ./object_detection_mono.py --hef yolov8m.hef -i usb --headless --track --show-fps

    # Custom labels file:
    ./object_detection_mono.py --hef yolov8m.hef -i usb --headless --labels my_labels.txt
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import collections
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from enum import Enum
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party (binary / pip-installed — not inlinable)
# ---------------------------------------------------------------------------
import cv2
import numpy as np
import scipy.linalg
import scipy.sparse

# Tracker dependencies
try:
    import lap
    from cython_bbox import bbox_overlaps as bbox_ious
    _TRACKER_AVAILABLE = True
except ImportError:
    _TRACKER_AVAILABLE = False

# Hailo runtime
from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm
from hailo_platform.pyhailort.pyhailort import FormatOrder


# ===========================================================================
# SECTION 1 — Constants
# ===========================================================================

MAX_INPUT_QUEUE_SIZE  = 60
MAX_OUTPUT_QUEUE_SIZE = 60
MAX_ASYNC_INFER_JOBS  = 20

CAMERA_RESOLUTION_MAP: Dict[str, Tuple[int, int]] = {
    "sd":  (640, 480),
    "hd":  (1280, 720),
    "fhd": (1920, 1080),
}

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# COCO 80-class labels (embedded so no external file is required by default)
_EMBEDDED_COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Default visualization config (mirrors config.json in the original repo)
_DEFAULT_CONFIG: Dict[str, Any] = {
    "visualization_params": {
        "score_thres": 0.25,
        "max_boxes_to_draw": 500,
        "tracker": {
            "track_thresh": 0.1,
            "track_buffer": 30,
            "match_thresh": 0.9,
            "aspect_ratio_thresh": 2.0,
            "min_box_area": 500,
            "mot20": False,
        },
    }
}

# Arducam 0234 native resolution for "scaledsd" mode
_SCALEDSD_CAP_W, _SCALEDSD_CAP_H = 960, 600
_SCALEDSD_OUT_W, _SCALEDSD_OUT_H = 640, 480

# Only draw motion trails for these COCO class IDs
_TRACKLET_CLASSES = [0, 67]  # person, cell phone
_TRAIL_LENGTH     = 30


# ===========================================================================
# SECTION 2 — Logger
# ===========================================================================

SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def _success(self, msg, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, msg, args, **kwargs)

logging.Logger.success = _success

_ANSI = {
    "DEBUG":   "\033[36m",
    "INFO":    "\033[0m",
    "SUCCESS": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR":   "\033[31m",
    "CRITICAL":"\033[35m",
}
_RST = "\033[0m"


class _ColorFormatter(logging.Formatter):
    _FMT = "%(levelname)s | %(name)s | %(message)s"

    def format(self, record):
        parts = record.name.split(".")
        record.name = ".".join(parts[-2:]) if len(parts) > 2 else record.name
        msg = logging.Formatter(self._FMT).format(record)
        if sys.stdout.isatty():
            color = _ANSI.get(record.levelname, "")
            msg = color + msg + _RST
        return msg


def _init_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_ColorFormatter())
    root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


logger = get_logger(__name__)


# ===========================================================================
# SECTION 3 — Kalman Filter
# ===========================================================================

chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877,
             5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}


class KalmanFilter:
    """Constant-velocity Kalman filter for 8-dimensional (x,y,a,h,vx,vy,va,vh) state."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros_like(measurement)]
        std = [2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3], 1e-2,
               2 * self._std_weight_position * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3], 1e-5,
               10 * self._std_weight_velocity * measurement[3]]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        std_pos = [self._std_weight_position * mean[3]] * 3 + [1e-2]
        std_vel = [self._std_weight_velocity * mean[3]] * 3 + [1e-5]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot(
            (self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3], 1e-1,
               self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        std_pos = [self._std_weight_position * mean[:, 3]] * 3 + \
                  [1e-2 * np.ones_like(mean[:, 3])]
        std_vel = [self._std_weight_velocity * mean[:, 3]] * 3 + \
                  [1e-5 * np.ones_like(mean[:, 3])]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = np.asarray([np.diag(sqr[i]) for i in range(len(mean))])
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose(1, 0, 2)
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_cov = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_cov


# ===========================================================================
# SECTION 4 — Tracker base types
# ===========================================================================

class TrackState:
    New = 0; Tracked = 1; Lost = 2; Removed = 3


class BaseTrack:
    _count = 0
    track_id = 0
    is_activated = False
    state = TrackState.New
    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args): raise NotImplementedError
    def predict(self):         raise NotImplementedError
    def update(self, *args, **kwargs): raise NotImplementedError
    def mark_lost(self):    self.state = TrackState.Lost
    def mark_removed(self): self.state = TrackState.Removed


# ===========================================================================
# SECTION 5 — Matching (requires lap + cython_bbox)
# ===========================================================================

class Matching:

    @staticmethod
    def linear_assignment(cost_matrix, thresh):
        if not _TRACKER_AVAILABLE:
            raise RuntimeError("lap and cython_bbox are required for tracking. "
                               "Run: pip install lap cython_bbox")
        if cost_matrix.size == 0:
            return (np.empty((0, 2), dtype=int),
                    tuple(range(cost_matrix.shape[0])),
                    tuple(range(cost_matrix.shape[1])))
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = np.asarray([[i, x[i]] for i in range(len(x)) if x[i] >= 0])
        return matches, np.where(x < 0)[0], np.where(y < 0)[0]

    @staticmethod
    def ious(atlbrs, btlbrs):
        ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
        if ious.size == 0:
            return ious
        return bbox_ious(
            np.ascontiguousarray(atlbrs, dtype=np.float64),
            np.ascontiguousarray(btlbrs, dtype=np.float64))

    @staticmethod
    def iou_distance(atracks, btracks):
        atlbrs = [t.tlbr if not isinstance(t, np.ndarray) else t for t in atracks]
        btlbrs = [t.tlbr if not isinstance(t, np.ndarray) else t for t in btracks]
        return 1 - Matching.ious(atlbrs, btlbrs)

    @staticmethod
    def fuse_score(cost_matrix, detections):
        if cost_matrix.size == 0:
            return cost_matrix
        iou_sim = 1 - cost_matrix
        det_scores = np.expand_dims(
            np.array([d.score for d in detections]), 0
        ).repeat(cost_matrix.shape[0], axis=0)
        return 1 - iou_sim * det_scores


# ===========================================================================
# SECTION 6 — STrack + BYTETracker
# ===========================================================================

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean = self.covariance = None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if not stracks:
            return
        multi_mean = np.asarray([s.mean.copy() for s in stracks])
        multi_cov  = np.asarray([s.covariance for s in stracks])
        for i, s in enumerate(stracks):
            if s.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_cov = STrack.shared_kalman.multi_predict(
            multi_mean, multi_cov)
        for i, (m, c) in enumerate(zip(multi_mean, multi_cov)):
            stracks[i].mean       = m
            stracks[i].covariance = c

    def activate(self, kf, frame_id):
        self.kalman_filter = kf
        self.track_id = self.next_id()
        self.mean, self.covariance = kf.initiate(
            self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = (frame_id == 1)
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self):
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks  = []
        self.lost_stracks     = []
        self.removed_stracks  = []
        self.frame_id         = 0
        self.args             = args
        self.det_thresh       = args.track_thresh + 0.1
        self.buffer_size      = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost    = self.buffer_size
        self.kalman_filter    = KalmanFilter()

    def update(self, output_results):
        self.frame_id += 1
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]

        dets        = bboxes[scores > self.args.track_thresh]
        scores_keep = scores[scores > self.args.track_thresh]
        dets_second = bboxes[(scores > 0.1) & (scores < self.args.track_thresh)]
        scores_sec  = scores[(scores > 0.1) & (scores < self.args.track_thresh)]

        detections  = [STrack(STrack.tlbr_to_tlwh(b), s)
                       for b, s in zip(dets, scores_keep)]
        det_second  = [STrack(STrack.tlbr_to_tlwh(b), s)
                       for b, s in zip(dets_second, scores_sec)]

        unconfirmed, tracked = [], []
        for t in self.tracked_stracks:
            (tracked if t.is_activated else unconfirmed).append(t)

        pool = _joint(tracked, self.lost_stracks)
        STrack.multi_predict(pool)

        # First association (high-score boxes)
        dists = Matching.iou_distance(pool, detections)
        if not self.args.mot20:
            dists = Matching.fuse_score(dists, detections)
        matches, u_track, u_det = Matching.linear_assignment(
            dists, self.args.match_thresh)
        activated, refind = [], []
        for it, id_ in matches:
            t, d = pool[it], detections[id_]
            if t.state == TrackState.Tracked:
                t.update(d, self.frame_id); activated.append(t)
            else:
                t.re_activate(d, self.frame_id); refind.append(t)

        # Second association (low-score boxes)
        r_tracked = [pool[i] for i in u_track
                     if pool[i].state == TrackState.Tracked]
        dists2 = Matching.iou_distance(r_tracked, det_second)
        matches2, u_track2, _ = Matching.linear_assignment(dists2, 0.5)
        lost = []
        for it, id_ in matches2:
            t, d = r_tracked[it], det_second[id_]
            if t.state == TrackState.Tracked:
                t.update(d, self.frame_id); activated.append(t)
            else:
                t.re_activate(d, self.frame_id); refind.append(t)
        for i in u_track2:
            t = r_tracked[i]
            if t.state != TrackState.Lost:
                t.mark_lost(); lost.append(t)

        # Unconfirmed tracks
        rem_dets = [detections[i] for i in u_det]
        dists3 = Matching.iou_distance(unconfirmed, rem_dets)
        if not self.args.mot20:
            dists3 = Matching.fuse_score(dists3, rem_dets)
        matches3, u_unconf, u_det3 = Matching.linear_assignment(dists3, 0.7)
        removed = []
        for it, id_ in matches3:
            unconfirmed[it].update(rem_dets[id_], self.frame_id)
            activated.append(unconfirmed[it])
        for i in u_unconf:
            unconfirmed[i].mark_removed(); removed.append(unconfirmed[i])
        for i in u_det3:
            t = rem_dets[i]
            if t.score >= self.det_thresh:
                t.activate(self.kalman_filter, self.frame_id)
                activated.append(t)

        # Update lost/removed lists
        for t in self.lost_stracks:
            if self.frame_id - t.end_frame > self.max_time_lost:
                t.mark_removed(); removed.append(t)

        self.tracked_stracks = _joint(
            [t for t in self.tracked_stracks if t.state == TrackState.Tracked],
            _joint(activated, refind))
        self.lost_stracks = _sub(
            _joint(self.lost_stracks, lost), self.tracked_stracks)
        self.lost_stracks = _sub(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = _remove_dups(
            self.tracked_stracks, self.lost_stracks)
        return [t for t in self.tracked_stracks if t.is_activated]


def _joint(a, b):
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen]

def _sub(a, b):
    rm = {t.track_id for t in b}
    return [t for t in a if t.track_id not in rm]

def _remove_dups(a, b):
    pdist = Matching.iou_distance(a, b)
    pairs = np.where(pdist < 0.15)
    da, db = set(), set()
    for p, q in zip(*pairs):
        ta = a[p].frame_id - a[p].start_frame
        tb = b[q].frame_id - b[q].start_frame
        (db if ta > tb else da).add(p if ta <= tb else q)
    return ([t for i, t in enumerate(a) if i not in da],
            [t for i, t in enumerate(b) if i not in db])


# ===========================================================================
# SECTION 7 — HailoInfer
# ===========================================================================

class HailoInfer:
    """Async inference wrapper for a single HEF model on the Hailo NPU."""

    def __init__(self, hef_path: str, batch_size: int = 1,
                 input_type: Optional[str] = None,
                 output_type: Optional[str] = None,
                 priority: int = 0) -> None:
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        self.target = VDevice(params)

        hef_path = os.fspath(hef_path)
        self.hef = HEF(hef_path)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)

        self._set_input_type(input_type)
        self._set_output_type(output_type)

        self.config_ctx      = self.infer_model.configure()
        self.configured_model = self.config_ctx.__enter__()
        self.configured_model.set_scheduler_priority(priority)
        self.last_infer_job  = None

    def _set_input_type(self, input_type):
        if input_type is not None:
            self.infer_model.input().set_format_type(
                getattr(FormatType, input_type))

    def _set_output_type(self, output_type):
        self.nms_postprocess_enabled = False
        if self.infer_model.outputs[0].format.order == FormatOrder.HAILO_NMS_WITH_BYTE_MASK:
            self.nms_postprocess_enabled = True
            self.output_type = self._type_dict("UINT8")
            return
        self.output_type = self._type_dict(output_type)
        for name, dtype in self.output_type.items():
            self.infer_model.output(name).set_format_type(
                getattr(FormatType, dtype))

    def _type_dict(self, data_type):
        valid = {"float32", "uint8", "uint16"}
        result = {}
        for info in self.hef.get_output_vstream_infos():
            name = info.name
            if data_type is None:
                result[name] = str(info.format.type).split(".")[-1]
            else:
                if data_type.lower() not in valid:
                    raise ValueError(f"Invalid data_type: {data_type}")
                result[name] = data_type
        return result

    def get_input_shape(self) -> Tuple[int, ...]:
        return self.hef.get_input_vstream_infos()[0].shape

    def run(self, input_batch: List[np.ndarray], callback_fn) -> object:
        bindings = self._create_bindings(self.configured_model, input_batch)
        self.configured_model.wait_for_async_ready(timeout_ms=10000)
        self.last_infer_job = self.configured_model.run_async(
            bindings,
            partial(callback_fn, bindings_list=bindings))
        return self.last_infer_job

    def _create_bindings(self, configured_model, input_batch):
        def _bind(frame):
            bufs = {
                name: np.empty(
                    self.infer_model.output(name).shape,
                    dtype=getattr(np, self.output_type[name].lower()))
                for name in self.output_type
            }
            b = configured_model.create_bindings(output_buffers=bufs)
            b.input().set_buffer(np.array(frame))
            return b
        return [_bind(f) for f in input_batch]

    def close(self):
        if self.last_infer_job is not None:
            self.last_infer_job.wait(10000)
        if self.config_ctx:
            self.config_ctx.__exit__(None, None, None)


# ===========================================================================
# SECTION 8 — Camera adapters
# ===========================================================================

class CroppedCapAdapter:
    """
    Wraps cv2.VideoCapture and center-crops every frame to (out_w, out_h).

    Used by "scaledsd" mode: the camera opens at its native 960x600
    (guaranteed by the Arducam 0234 driver), and each frame is cropped to
    640x480 in software — no scaling, no letterboxing, no geometry distortion.

    Crop offsets for 960x600 -> 640x480:
        x0 = (960 - 640) // 2 = 160
        y0 = (600 - 480) // 2 =  60
    """
    def __init__(self, cap: cv2.VideoCapture, out_w: int, out_h: int):
        self._cap   = cap
        self._out_w = out_w
        self._out_h = out_h
        cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or out_w)
        cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or out_h)
        self._x0 = (cap_w - out_w) // 2
        self._y0 = (cap_h - out_h) // 2

    def isOpened(self):  return self._cap.isOpened()
    def release(self):   self._cap.release()

    def read(self):
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame[
            self._y0:self._y0 + self._out_h,
            self._x0:self._x0 + self._out_w,
        ]

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:  return float(self._out_w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return float(self._out_h)
        return self._cap.get(prop_id)


class PiCamera2CaptureAdapter:
    """Makes Picamera2 behave like cv2.VideoCapture (thread-safe release)."""
    def __init__(self, picam2):
        self.picam2    = picam2
        self._opened   = True
        self._io_lock  = threading.Lock()

    def isOpened(self):   return self._opened

    def read(self):
        if not self._opened:
            return False, None
        with self._io_lock:
            if not self._opened:
                return False, None
            frame = self.picam2.capture_array()
        return (False, None) if frame is None else (True, frame)

    def get(self, prop_id: int) -> float:
        if prop_id in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            try:
                cfg  = self.picam2.camera_configuration()
                size = cfg.get("main", {}).get("size")
                if size and len(size) == 2:
                    w, h = int(size[0]), int(size[1])
                    return float(w if prop_id == cv2.CAP_PROP_FRAME_WIDTH else h)
            except Exception:
                pass
            return 0.0
        if prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def release(self):
        self._opened = False
        with self._io_lock:
            for fn in (self.picam2.stop, self.picam2.close):
                try: fn()
                except Exception: pass


# ===========================================================================
# SECTION 9 — Capture processing mode
# ===========================================================================

class CapProcessingMode(str, Enum):
    CAMERA_NORMAL     = "camera_normal"
    CAMERA_FRAME_DROP = "camera_frame_drop"
    VIDEO_NORMAL      = "video_normal"
    VIDEO_PACE        = "video_pace"


# ===========================================================================
# SECTION 10 — Camera / input utilities
# ===========================================================================

def is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model") as f:
            return "Raspberry Pi" in f.read()
    except Exception:
        return False


def is_stream_url(s: str) -> bool:
    return s.startswith(("http://", "https://", "rtsp://"))


def get_usb_video_devices() -> Dict[int, str]:
    """Return {video_index: device_name} for USB-backed V4L2 nodes only."""
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"],
            stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        logger.error(f"Failed to run v4l2-ctl --list-devices: {e}")
        return {}

    devices: Dict[int, str] = {}
    header = ""
    is_usb = False

    for line in out.splitlines():
        if not line.strip():
            continue
        if not line.startswith("\t"):
            header = line.strip().rstrip(":")
            lower  = header.lower()
            is_usb = (bool(re.search(r"\([0-9a-f]{4}:[0-9a-f]{4}\)", header, re.I))
                      or "(usb-" in lower or " usb-" in lower or "(usb:" in lower)
            continue
        if is_usb and "/dev/video" in line:
            m = re.search(r"/dev/video(\d+)", line)
            if m:
                devices[int(m.group(1))] = header
    return devices


def open_usb_camera(resolution: Optional[str]):
    """
    Open a USB camera via V4L2.

    resolution options:
        None / "native" — let driver decide (no cap.set calls)
        "sd"             — request 640x480
        "hd"             — request 1280x720
        "fhd"            — request 1920x1080
        "scaledsd"       — open at native 960x600, then crop to 640x480
    """
    usb_devices = get_usb_video_devices()
    if not usb_devices:
        logger.error("USB mode requested but NO USB cameras detected.")
        logger.error("Run: v4l2-ctl --list-devices")
        sys.exit(1)

    env_val = os.environ.get("CAMERA_INDEX")
    if env_val is None:
        camera_index = sorted(usb_devices.keys())[0]
        logger.debug(f"Auto-selected USB camera index {camera_index} "
                     f"({usb_devices[camera_index]})")
    else:
        try:
            camera_index = int(env_val)
        except ValueError:
            logger.error(f"Invalid CAMERA_INDEX value: {env_val}")
            sys.exit(1)
        if camera_index not in usb_devices:
            logger.error(f"CAMERA_INDEX={camera_index} is not a USB camera. "
                         f"Available: {sorted(usb_devices.keys())}")
            sys.exit(1)

    # Force V4L2 backend — avoids GStreamer YUYV issues in headless mode
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        logger.error(f"Failed to open USB camera index {camera_index}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Resolution handling
    # -----------------------------------------------------------------------
    if resolution == "scaledsd":
        # Request native resolution so the driver never has to negotiate
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  _SCALEDSD_CAP_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _SCALEDSD_CAP_H)
        logger.debug(f"scaledsd: requesting {_SCALEDSD_CAP_W}x{_SCALEDSD_CAP_H}, "
                     f"will crop to {_SCALEDSD_OUT_W}x{_SCALEDSD_OUT_H}")
    elif resolution is not None and resolution != "native" \
            and resolution in CAMERA_RESOLUTION_MAP:
        w, h = CAMERA_RESOLUTION_MAP[resolution]
        rw = cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        rh = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if not rw or not rh:
            logger.warning(f"Driver rejected {w}x{h} — will use nearest supported mode.")
        else:
            logger.debug(f"USB camera resolution requested: {w}x{h}")

    # Validate stream — also logs what the driver actually gave us
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        logger.error("USB camera opened but produced no frames.")
        sys.exit(1)

    ah, aw = frame.shape[:2]
    logger.info(f"USB camera streaming at {aw}x{ah} (YUYV→BGR via V4L2)")

    # For scaledsd: validate then wrap in cropping adapter
    if resolution == "scaledsd":
        if aw < _SCALEDSD_OUT_W or ah < _SCALEDSD_OUT_H:
            cap.release()
            logger.error(f"scaledsd: camera delivered {aw}x{ah}, smaller than "
                         f"crop target {_SCALEDSD_OUT_W}x{_SCALEDSD_OUT_H}.")
            sys.exit(1)
        adapter = CroppedCapAdapter(cap, _SCALEDSD_OUT_W, _SCALEDSD_OUT_H)
        logger.info(f"scaledsd: center-cropping {aw}x{ah} → "
                    f"{_SCALEDSD_OUT_W}x{_SCALEDSD_OUT_H} (no scaling)")
        return adapter

    return cap


def open_rpi_camera():
    try:
        from picamera2 import Picamera2
    except Exception as e:
        logger.error(f"Picamera2 not available: {e}")
        return None
    try:
        picam2 = Picamera2()
        cfg = picam2.create_video_configuration(
            main={"size": (800, 600), "format": "RGB888"},
            controls={"FrameRate": 30})
        picam2.configure(cfg)
        picam2.start()
        return PiCamera2CaptureAdapter(picam2)
    except Exception as e:
        logger.error(f"Failed to open RPi camera: {e}")
        for fn in (picam2.stop, picam2.close):
            try: fn()
            except Exception: pass
        return None


def init_input_source(input_src: str, batch_size: int, resolution: Optional[str]):
    """
    Resolve the input source string to (cap, images, input_type).

    input_src values:
        "usb"           — USB camera
        "rpi"           — Raspberry Pi CSI camera
        http/https/rtsp — network stream
        *.mp4 / *.avi … — video file
        file/folder     — image(s)
    """
    src = input_src.strip()

    if src == "usb":
        cap = open_usb_camera(resolution)
        logger.info("Using USB camera")
        return cap, None, "usb"

    if src == "rpi":
        if not is_raspberry_pi():
            logger.error("RPi camera requested but this is not a Raspberry Pi.")
            sys.exit(1)
        cap = open_rpi_camera()
        if cap is None:
            sys.exit(1)
        logger.info("Using Raspberry Pi camera at 800x600")
        return cap, None, "rpi"

    if is_stream_url(src):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error(f"Failed to open stream: {src}")
            sys.exit(1)
        logger.info(f"Using stream: {src}")
        return cap, None, "stream"

    if any(src.lower().endswith(s) for s in VIDEO_SUFFIXES):
        if not os.path.exists(src):
            logger.error(f"File not found: {src}")
            sys.exit(1)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {src}")
            sys.exit(1)
        logger.info(f"Using video file: {src}")
        return cap, None, "video"

    if not os.path.exists(src):
        logger.error(f"Invalid input '{src}'. Expected: usb | rpi | url | video | image path")
        sys.exit(1)

    images = _load_images(src)
    if not images:
        logger.error(f"No valid images found in: {src}")
        sys.exit(1)
    if len(images) % batch_size != 0:
        logger.error(f"Image count ({len(images)}) not divisible by batch_size ({batch_size})")
        sys.exit(1)
    return None, images, "images"


def _load_images(path_str: str) -> List[np.ndarray]:
    path = Path(path_str)

    def read_rgb(p):
        img = cv2.imread(str(p))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        img = read_rgb(path)
        return [img] if img is not None else []
    if path.is_dir():
        imgs = [read_rgb(p) for p in path.glob("*")
                if p.suffix.lower() in IMAGE_EXTENSIONS]
        return [i for i in imgs if i is not None]
    return []


# ===========================================================================
# SECTION 11 — Preprocessing
# ===========================================================================

def select_cap_processing_mode(input_type: str, save_output: bool,
                                frame_rate: Optional[float]) -> CapProcessingMode:
    is_camera = input_type in ("usb", "rpi", "stream")
    has_fps   = frame_rate is not None and frame_rate > 0
    if is_camera:
        return (CapProcessingMode.CAMERA_FRAME_DROP if has_fps
                else CapProcessingMode.CAMERA_NORMAL)
    if input_type == "video":
        return (CapProcessingMode.VIDEO_PACE if save_output
                else CapProcessingMode.VIDEO_NORMAL)
    return None


def default_preprocess(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    """Letterbox-pad image to model input size preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = min(model_w / w, model_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    out = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
    xo = (model_w - nw) // 2
    yo = (model_h - nh) // 2
    out[yo:yo + nh, xo:xo + nw] = resized
    return out


def preprocess(images, cap, framerate, batch_size, input_queue,
               width, height, cap_processing_mode,
               preprocess_fn=None, stop_event=None):
    preprocess_fn = preprocess_fn or default_preprocess
    if cap is None:
        _preprocess_images(images, batch_size, input_queue, width, height, preprocess_fn)
    else:
        _preprocess_from_cap(cap, batch_size, input_queue, width, height,
                             cap_processing_mode, preprocess_fn, framerate, stop_event)
    # Single sentinel owned by this function only
    input_queue.put(None)


def _preprocess_images(images, batch_size, input_queue, width, height, preprocess_fn):
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        input_queue.put(
            ([img for img in batch],
             [preprocess_fn(img, width, height) for img in batch]))


def _preprocess_from_cap(cap, batch_size, input_queue, width, height,
                          mode, preprocess_fn, target_fps, stop_event):
    def should_stop():
        return stop_event is not None and stop_event.is_set()

    next_ts    = time.monotonic()
    keep_period = (1.0 / float(target_fps)) if mode == CapProcessingMode.CAMERA_FRAME_DROP else None
    vt0 = wt0 = None
    frames, processed = [], []

    while not should_stop():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if mode == CapProcessingMode.VIDEO_PACE:
            pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if vt0 is None:
                vt0 = pos_ms; wt0 = time.monotonic()
            desired = wt0 + (pos_ms - vt0) / 1000.0
            now = time.monotonic()
            if now < desired:
                time.sleep(desired - now)

        if mode == CapProcessingMode.CAMERA_FRAME_DROP:
            now = time.monotonic()
            if now < next_ts:
                continue
            next_ts += keep_period

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        processed.append(preprocess_fn(frame_rgb, width, height))

        if len(frames) >= batch_size:
            input_queue.put((frames, processed))
            frames, processed = [], []

    # Flush partial last batch
    if frames and not should_stop():
        input_queue.put((frames, processed))
    # Sentinel written by caller (preprocess), not here — avoids double-sentinel


# ===========================================================================
# SECTION 12 — Post-process (detection drawing)
# ===========================================================================

def _id_to_color(idx: int) -> np.ndarray:
    np.random.seed(idx)
    return np.random.randint(0, 255, size=3, dtype=np.uint8)


def _compute_iou(a, b) -> float:
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    aA = max(1e-5, (a[2] - a[0]) * (a[3] - a[1]))
    aB = max(1e-5, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aA + aB - inter + 1e-5)


def _best_match(track_box, det_boxes):
    best_iou, best_idx = 0, -1
    for i, db in enumerate(det_boxes):
        iou = _compute_iou(track_box, db)
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return best_idx if best_idx != -1 else None


def _denorm_rm_pad(box, size, pad, img_h, img_w):
    box = [int(x * size) for x in box]
    for i in range(4):
        if i % 2 == 0 and img_h != size: box[i] -= pad
        if i % 2 == 1 and img_w != size: box[i] -= pad
    return [box[1], box[0], box[3], box[2]]  # → [ymin, xmin, ymax, xmax]


def _extract_detections(image, raw, score_thres, max_boxes):
    h, w = image.shape[:2]
    size = max(h, w)
    pad  = int(abs(h - w) / 2)
    all_dets = []
    for cls_id, class_dets in enumerate(raw):
        for det in class_dets:
            bbox, score = det[:4], det[4]
            if score >= score_thres:
                box = _denorm_rm_pad(bbox, size, pad, h, w)
                all_dets.append((score, cls_id, box))
    all_dets.sort(reverse=True, key=lambda x: x[0])
    top = all_dets[:max_boxes]
    if top:
        scores, classes, boxes = zip(*top)
    else:
        scores, classes, boxes = [], [], []
    return {"detection_boxes": list(boxes),
            "detection_classes": list(classes),
            "detection_scores": list(scores),
            "num_detections": len(top)}


def _draw_one(image, box, label_lines, score, color, track=False):
    xmin, ymin, xmax, ymax = map(int, box)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    top  = f"{label_lines[0]}: {score:.1f}%" if not track or len(label_lines) == 2 else f"{score:.1f}%"
    bot  = label_lines[1] if (track and len(label_lines) == 2) else (label_lines[0] if track else None)
    for txt, pos in [(top, (xmin + 4, ymin + 20))]:
        cv2.putText(image, txt, pos, font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, txt, pos, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if bot:
        p = (xmax - 50, ymax - 6)
        cv2.putText(image, bot, p, font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, bot, p, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


_tracklet_history: Dict[int, deque] = {}


def inference_result_handler(original_frame, infer_results, labels,
                              config_data, tracker=None, draw_trail=False):
    """Full post-process: extract → draw (with optional tracking)."""
    vp           = config_data["visualization_params"]
    score_thres  = vp.get("score_thres", 0.25)
    max_boxes    = vp.get("max_boxes_to_draw", 500)
    dets         = _extract_detections(original_frame, infer_results,
                                       score_thres, max_boxes)
    img          = original_frame.copy()
    boxes        = dets["detection_boxes"]
    scores       = dets["detection_scores"]
    classes      = dets["detection_classes"]
    n            = dets["num_detections"]

    if tracker:
        if not n:
            return img
        raw = np.array([[*b, s] for b, s in zip(boxes, scores)])
        for track in tracker.update(raw):
            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.tlbr)
            bidx = _best_match(track.tlbr, boxes)
            color = tuple(_id_to_color(classes[bidx] if bidx is not None else 0).tolist())
            if bidx is not None:
                _draw_one(img, [x1, y1, x2, y2],
                          [labels[classes[bidx]], f"ID {tid}"],
                          track.score * 100, color, track=True)
                if classes[bidx] in _TRACKLET_CLASSES:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    _tracklet_history.setdefault(tid, deque(maxlen=_TRAIL_LENGTH))
                    _tracklet_history[tid].append((cx, cy))
                    if draw_trail:
                        hist = _tracklet_history[tid]
                        for i in range(1, len(hist)):
                            cv2.line(img, hist[i - 1], hist[i], color, 3)
                            cv2.circle(img, hist[i], 20, color, 1)
            else:
                _draw_one(img, [x1, y1, x2, y2], [f"ID {tid}"],
                          track.score * 100, color, track=True)
    else:
        for i in range(n):
            color = tuple(_id_to_color(classes[i]).tolist())
            _draw_one(img, boxes[i], [labels[classes[i]]],
                      scores[i] * 100, color)
    return img


# ===========================================================================
# SECTION 13 — Frame rate tracker
# ===========================================================================

class FrameRateTracker:
    def __init__(self):
        self._count = 0
        self._start = None

    def start(self):
        self._start = time.time()

    def increment(self, n=1):
        self._count += n

    @property
    def fps(self) -> float:
        if self._start is None:
            return 0.0
        e = time.time() - self._start
        return self._count / e if e > 0 else 0.0

    def summary(self) -> str:
        e = (time.time() - self._start) if self._start else 0.0
        return (f"Processed {self._count} frames at {self.fps:.2f} FPS, "
                f"total time: {e:.2f}s")


# ===========================================================================
# SECTION 14 — Visualize (headless / display switch)
# ===========================================================================

def _resize_for_output(frame, resolution):
    if resolution is None:
        return frame
    _, th = resolution
    h, w  = frame.shape[:2]
    if not h or not w:
        return frame
    scale = th / float(h)
    return cv2.resize(frame, (int(round(w * scale)), th),
                      interpolation=cv2.INTER_AREA)


def visualize(output_queue: queue.Queue, cap, save_output: bool,
              output_dir: str, callback: Callable,
              fps_tracker: Optional[FrameRateTracker] = None,
              output_resolution=None, framerate: Optional[float] = None,
              side_by_side: bool = False,
              stop_event: Optional[threading.Event] = None,
              headless: bool = True) -> None:
    """
    Consume output_queue, draw detections, and either:
      headless=True  — print FPS to terminal, save video/images to disk
      headless=False — show an OpenCV window; press 'q' (or Ctrl+C) to stop
    """
    image_id     = 0
    out          = None
    frame_width  = None
    frame_height = None

    if cap is not None:
        base_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
        base_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        target_w, target_h = (output_resolution
                              if output_resolution is not None
                              else (base_w, base_h))
        frame_width  = target_w * (2 if side_by_side else 1)
        frame_height = target_h

        if not headless:
            cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)

        if save_output:
            cam_fps   = cap.get(cv2.CAP_PROP_FPS)
            final_fps = framerate or (cam_fps if cam_fps and cam_fps > 1 else 30.0)
            os.makedirs(output_dir, exist_ok=True)
            out_path  = os.path.join(output_dir, "output.avi")
            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                final_fps,
                (frame_width, frame_height))

    quitting = False

    while True:
        result = output_queue.get()
        try:
            if result is None:
                break

            original_frame, inference_result, *meta = result

            if quitting:
                continue

            if isinstance(inference_result, list) and len(inference_result) == 1:
                inference_result = inference_result[0]

            frame_out = (callback(original_frame, inference_result, meta[0])
                         if meta else callback(original_frame, inference_result))

            if fps_tracker is not None:
                fps_tracker.increment()
                n_det = len(inference_result) if inference_result is not None else 0
                print(f"\rFPS: {fps_tracker.fps:.2f} | Detections: {n_det}",
                      end="", flush=True)

            bgr_frame   = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
            frame_show  = _resize_for_output(bgr_frame, output_resolution)

            if cap is not None:
                if not headless:
                    cv2.imshow("Output", frame_show)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        quitting = True
                        if stop_event is not None:
                            stop_event.set()
                        continue

                if save_output and out is not None and frame_width and frame_height:
                    out.write(cv2.resize(frame_show, (frame_width, frame_height)))
            else:
                # Image mode — always save
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(output_dir, f"output_{image_id}.png"),
                    frame_show)

            image_id += 1

        finally:
            output_queue.task_done()

    # Cleanup
    if out is not None:
        out.release()
    if cap is not None:
        cap.release()
    if fps_tracker is not None:
        print()  # newline after \r FPS line
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


# ===========================================================================
# SECTION 15 — Inference loop
# ===========================================================================

def _inference_callback(completion_info, bindings_list, input_batch, output_queue):
    if completion_info.exception:
        logger.error(f"Inference error: {completion_info.exception}")
        return
    for i, bindings in enumerate(bindings_list):
        if len(bindings._output_names) == 1:
            result = bindings.output().get_buffer()
        else:
            result = {
                name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                for name in bindings._output_names
            }
        output_queue.put((input_batch[i], result))


def infer(hailo_inference: HailoInfer, input_queue: queue.Queue,
          output_queue: queue.Queue, stop_event: threading.Event):
    pending = collections.deque()
    while True:
        batch = input_queue.get()
        if batch is None:
            break
        if stop_event.is_set():
            continue
        input_batch, preprocessed_batch = batch
        cb = partial(_inference_callback,
                     input_batch=input_batch, output_queue=output_queue)
        while len(pending) >= MAX_ASYNC_INFER_JOBS:
            pending.popleft().wait(10000)
        pending.append(hailo_inference.run(preprocessed_batch, cb))
    hailo_inference.close()
    output_queue.put(None)


# ===========================================================================
# SECTION 16 — Top-level pipeline
# ===========================================================================

def get_labels(labels_path: Optional[str]) -> List[str]:
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, encoding="utf-8") as f:
            return f.read().splitlines()
    return list(_EMBEDDED_COCO_LABELS)


def run_inference_pipeline(hef_path: str, input_src: str, batch_size: int,
                           labels_path: Optional[str], output_dir: str,
                           save_output: bool = False,
                           camera_resolution: Optional[str] = None,
                           output_resolution=None,
                           enable_tracking: bool = False,
                           show_fps: bool = False,
                           frame_rate: Optional[float] = None,
                           draw_trail: bool = False,
                           headless: bool = True) -> None:
    labels      = get_labels(labels_path)
    config_data = _DEFAULT_CONFIG

    cap, images, input_type = init_input_source(input_src, batch_size, camera_resolution)
    cap_mode = select_cap_processing_mode(input_type, save_output, frame_rate) \
               if cap is not None else None

    stop_event  = threading.Event()

    # SIGINT → gracefully set stop_event (so preprocess loop exits)
    _prev_sigint = signal.getsignal(signal.SIGINT)
    def _sigint(sig, frame):
        logger.info("\nStopping pipeline gracefully...")
        stop_event.set()
        signal.signal(signal.SIGINT, _prev_sigint)
    signal.signal(signal.SIGINT, _sigint)

    tracker     = None
    fps_tracker = FrameRateTracker() if show_fps else None

    if enable_tracking:
        if not _TRACKER_AVAILABLE:
            logger.error("Tracking requires: pip install lap cython_bbox")
            sys.exit(1)
        tcfg    = config_data["visualization_params"]["tracker"]
        tracker = BYTETracker(SimpleNamespace(**tcfg))

    input_queue  = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)

    callback = partial(inference_result_handler,
                       labels=labels, config_data=config_data,
                       tracker=tracker, draw_trail=draw_trail)

    hailo_inference = HailoInfer(hef_path, batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    t_pre  = threading.Thread(
        target=preprocess,
        args=(images, cap, frame_rate, batch_size, input_queue,
              width, height, cap_mode, None, stop_event))
    t_inf  = threading.Thread(
        target=infer,
        args=(hailo_inference, input_queue, output_queue, stop_event))
    t_post = threading.Thread(
        target=visualize,
        args=(output_queue, cap, save_output, output_dir,
              callback, fps_tracker, output_resolution,
              frame_rate, False, stop_event, headless))

    if show_fps:
        fps_tracker.start()

    t_pre.start(); t_inf.start(); t_post.start()

    t_pre.join(); t_inf.join(); t_post.join()

    signal.signal(signal.SIGINT, _prev_sigint)

    if show_fps:
        logger.info(fps_tracker.summary())

    logger.success("Inference was successful!")
    if save_output or input_src.lower() not in ("usb", "rpi"):
        logger.info(f"Results saved in: {output_dir}")


# ===========================================================================
# SECTION 17 — CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Hailo object detection — self-contained monolithic script.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  Static image:
    ./object_detection_mono.py --hef yolov8m.hef -i photo.png

  USB camera, headless (Ctrl+C to stop):
    ./object_detection_mono.py --hef yolov8m.hef -i usb --headless

  USB camera with display window ('q' to quit):
    ./object_detection_mono.py --hef yolov8m.hef -i usb

  USB camera, scaledsd (960x600 native → 640x480 crop), headless:
    ./object_detection_mono.py --hef yolov8m.hef -i usb -cr scaledsd --headless

  Save output video while running:
    ./object_detection_mono.py --hef yolov8m.hef -i usb --headless --save-output

  With tracking and FPS counter:
    ./object_detection_mono.py --hef yolov8m.hef -i usb --headless --track --show-fps
""")

    p.add_argument("--hef", "-n", required=True,
                   help="Path to the .hef model file (e.g. yolov8m.hef)")
    p.add_argument("--input", "-i", required=True,
                   help="Input source: 'usb', 'rpi', image/video path, or stream URL")
    p.add_argument("--headless", action="store_true",
                   help="Run without a display window. "
                        "Required when DISPLAY is not set (e.g. SSH, Ctrl+C to stop).")
    p.add_argument("--batch-size", "-b", type=int, default=1,
                   help="Inference batch size (default: 1)")
    p.add_argument("--labels", "-l", type=str, default=None,
                   help="Path to labels .txt file (one label per line). "
                        "Defaults to embedded COCO 80-class labels.")
    p.add_argument("--output-dir", "-o", type=str, default=None,
                   help="Directory for saved output frames/video. "
                        "Defaults to ./output/")
    p.add_argument("--save-output", action="store_true",
                   help="Save annotated frames to disk. "
                        "Images → PNG files; camera/video → output.avi")
    p.add_argument("--camera-resolution", "-cr", type=str, default=None,
                   choices=["sd", "hd", "fhd", "scaledsd", "native"],
                   help=("Camera capture resolution preset:\n"
                         "  sd       640x480 (advisory)\n"
                         "  hd       1280x720 (advisory)\n"
                         "  fhd      1920x1080 (advisory)\n"
                         "  scaledsd capture at native 960x600, center-crop to 640x480\n"
                         "           (guaranteed, no scaling — recommended for Arducam 0234)\n"
                         "  native   let the driver decide (default)"))
    p.add_argument("--output-resolution", "-or", type=int, nargs=2,
                   metavar=("WIDTH", "HEIGHT"),
                   help="Resize output frames before saving/display (e.g. 1280 720)")
    p.add_argument("--frame-rate", "-f", type=float, default=None,
                   help="Target FPS for camera frame-drop mode (default: no drop)")
    p.add_argument("--show-fps", action="store_true",
                   help="Print FPS and detection count to terminal")
    p.add_argument("--track", action="store_true",
                   help="Enable BYTETrack object tracking (requires lap + cython_bbox)")
    p.add_argument("--draw-trail", action="store_true",
                   help="[tracking only] Draw motion trails for tracked objects")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: INFO)")
    return p.parse_args()


# ===========================================================================
# SECTION 18 — Entry point
# ===========================================================================

def main():
    args = parse_args()
    _init_logging(args.log_level)

    # Validate HEF path (user supplies it directly — no auto-download)
    if not os.path.isfile(args.hef):
        logger.error(f"HEF file not found: {args.hef}")
        sys.exit(1)

    # Validate output dir
    output_dir = args.output_dir or os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Auto-headless when no DISPLAY is available (safety fallback)
    headless = args.headless
    if not headless and not os.environ.get("DISPLAY") and \
            os.environ.get("QT_QPA_PLATFORM", "") != "offscreen":
        logger.warning("No DISPLAY detected — forcing --headless mode.")
        headless = True

    out_res = tuple(args.output_resolution) if args.output_resolution else None

    run_inference_pipeline(
        hef_path          = args.hef,
        input_src         = args.input,
        batch_size        = args.batch_size,
        labels_path       = args.labels,
        output_dir        = output_dir,
        save_output       = args.save_output,
        camera_resolution = args.camera_resolution,
        output_resolution = out_res,
        enable_tracking   = args.track,
        show_fps          = args.show_fps,
        frame_rate        = args.frame_rate,
        draw_trail        = args.draw_trail,
        headless          = headless,
    )


if __name__ == "__main__":
    main()
