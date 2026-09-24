#!/usr/bin/env python3
"""
Hand Gesture Laptop Controller (Modular Entry Point)
Lightweight real-time controller using MediaPipe Hands and trained Machine Learning model.
Supports local webcam (0) or smartphone IP camera stream URL (HTTP/RTSP).
"""

import sys
import time
import cv2
import joblib
import mediapipe as mp
import pyautogui
from pynput.mouse import Controller

from src.utils import extract_features
from src.smoother import GestureSmoother
from src.actions import ActionEngine

# Load model Machine Learning & pemetaan kelas
MODEL_FILE = "best_gesture_model.joblib"
checkpoint = joblib.load(MODEL_FILE)
model, class_map = checkpoint["model"], checkpoint["class_map"]

def main():
    mouse = Controller()
    try:
        sw, sh = pyautogui.size()
    except Exception:
        sw, sh = 1680, 1050

    # Sumber kamera: 0 (webcam bawaan) atau URL IP kamera HP (contoh: http://192.168.1.15:8080/video)
    cam_source = sys.argv[1] if len(sys.argv) > 1 else 0
    try:
        cam_source = int(cam_source)
    except ValueError:
        pass

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, model_complexity=0)

    print(f"[INFO] Membuka kamera: {cam_source} ...")
    cap = cv2.VideoCapture(cam_source)

    if isinstance(cam_source, str):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"Error: Kamera/stream '{cam_source}' tidak dapat dibuka.")
        return

    smoother = GestureSmoother(model.classes_)
    actions = ActionEngine(mouse, sw, sh)
    prev_time = time.time()
    fps = 0.0

    print(f"[INFO] Model aktif: {checkpoint.get('model_name')} pada layar {sw}x{sh}.")
    print("[INFO] Tekan 'q' pada jendela kamera untuk keluar.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 0.001))
        prev_time = curr_time

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mode = "NO HAND"
        candidate_mode = "-"
        confidence = 0.0

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            # Stabilkan probabilitas prediksi model
            stable_label, candidate_label, confidence = smoother.update(
                model.predict_proba([extract_features(lm)])
            )
            candidate_mode = class_map.get(candidate_label, "UNKNOWN")
            mode = class_map.get(stable_label, "STANDBY")
            actions.dispatch(mode, lm, w, h)
        else:
            smoother.reset()
            actions.dispatch(mode, None, w, h)

        # Gambar batas area kontrol & status HUD
        cv2.rectangle(frame, (100, 80), (w - 100, h - 80), (80, 80, 80), 1)
        cv2.putText(frame, f"MODE: {mode}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"MODEL: {candidate_mode} ({confidence:.0%})", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Gesture Controller", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
