#!/usr/bin/env python3
"""
face_detect.py

Deteksi wajah dari kamera laptop — dukung dua metode:
 - haar  : Haar Cascade (OpenCV)
 - dnn   : OpenCV DNN (res10_300x300_ssd)

Usage:
    python face_detect.py --model haar
    python face_detect.py --model dnn

Tombol:
  s : simpan snapshot (folder snapshots/)
  q : keluar
"""

import cv2
import time
import os
import argparse
import sys

# Optional: untuk mengunduh model DNN jika tidak ada
try:
    import requests
except Exception:
    requests = None

# ---- Arg parsing ----
parser = argparse.ArgumentParser(description="Face detection from webcam (haar or dnn)")
parser.add_argument("--model", choices=["haar", "dnn"], default="haar", help="Detection model to use")
parser.add_argument("--camera", type=int, default=0, help="Camera device index (default 0)")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for DNN (0-1)")
args = parser.parse_args()

# ---- Paths & download helper for DNN ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

# DNN model files (OpenCV res10 SSD)
PROTOTXT = os.path.join(BASE_DIR, "deploy.prototxt.txt")
CAFFEMODEL = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

DNN_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_CAFFE_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/res10_300x300_ssd_iter_140000.caffemodel"

def download_file(url, dest):
    if requests is None:
        print("requests library not available — cannot auto-download models. Please install 'requests' or place model files manually.")
        return False
    print(f"Downloading {url} → {dest} ...")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print("Download failed:", r.status_code)
        return False
    total = r.headers.get('content-length')
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return True

# ---- Initialize chosen detector ----
detector = None
if args.model == "haar":
    # use OpenCV's built-in haarcascade (no external file required)
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(haar_path):
        print("Error: Haar cascade XML not found at", haar_path)
        sys.exit(1)
    detector = cv2.CascadeClassifier(haar_path)
    print("Using Haar Cascade:", haar_path)

else:
    # DNN model: ensure files exist or try to download
    if not os.path.exists(PROTOTXT):
        ok = download_file(DNN_PROTOTXT_URL, PROTOTXT)
        if not ok:
            print("Cannot obtain prototxt. Place it manually at", PROTOTXT)
            sys.exit(1)
    if not os.path.exists(CAFFEMODEL):
        ok = download_file(DNN_CAFFE_URL, CAFFEMODEL)
        if not ok:
            print("Cannot obtain caffemodel. Place it manually at", CAFFEMODEL)
            sys.exit(1)
    print("Loading DNN model...")
    detector = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
    # optionally prefer CPU/GPU backend — default CPU
    # detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("DNN model loaded.")

# ---- Open camera ----
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print("Cannot open camera index", args.camera)
    sys.exit(1)

print("Starting camera. Press 's' to snapshot, 'q' to quit.")
fps_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    h, w = frame.shape[:2]
    orig = frame.copy()

    boxes = []

    if args.model == "haar":
        # Haar expects grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # faces -> list of (x,y,w,h)
        for (x, y, fw, fh) in faces:
            boxes.append((x, y, x+fw, y+fh, 1.0))  # dummy conf 1.0

    else:
        # DNN: create blob and forward
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        detector.setInput(blob)
        detections = detector.forward()
        # detections shape: [1, 1, N, 7] where last 7 elements: [0, 1, conf, x1, y1, x2, y2]
        for i in range(0, detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < args.conf:
                continue
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            # clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            boxes.append((x1, y1, x2, y2, conf))

    # draw boxes
    for (x1, y1, x2, y2, conf) in boxes:
        label = f"face: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

    # FPS counter
    if frame_count >= 10:
        now = time.time()
        fps = frame_count / (now - fps_time)
        fps_time = now
        frame_count = 0
    else:
        fps = None

    status_text = f"Model: {args.model.upper()}  Faces: {len(boxes)}"
    if fps:
        status_text += f"  FPS: {fps:.1f}"
    cv2.putText(frame, status_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        # save snapshot
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = os.path.join(SNAP_DIR, f"snapshot_{ts}.jpg")
        cv2.imwrite(fname, orig)
        print("Snapshot saved:", fname)

# cleanup
cap.release()
cv2.destroyAllWindows()
