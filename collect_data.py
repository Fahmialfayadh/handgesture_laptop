#!/usr/bin/env python3
"""
Data Collector untuk Klasifikasi Gestur Tangan
Merekam 42 koordinat normalisasi sendi tangan langsung ke dataset_gestures.csv
"""

import os
import math
import cv2
import pandas as pd
import mediapipe as mp

CLASS_MAP = {
    0: "STANDBY",
    1: "MOVE",
    2: "SCROLL",
    3: "LEFT_CLICK",
    4: "RIGHT_CLICK"
}

CSV_FILE = "dataset_gestures.csv"

def extract_features(landmarks):
    """Normalisasi 21 koordinat terhadap pergelangan tangan dan skala telapak."""
    base_x = landmarks[0].x
    base_y = landmarks[0].y

    raw_diffs = []
    for lm in landmarks:
        raw_diffs.append(lm.x - base_x)
        raw_diffs.append(lm.y - base_y)

    palm_size = math.hypot(landmarks[9].x - base_x, landmarks[9].y - base_y)
    if palm_size < 1e-6:
        palm_size = 1.0

    return [val / palm_size for val in raw_diffs]

def main():
    # Ambil sumber kamera dari argumen terminal jika ada
    import sys
    cam_source = sys.argv[1] if len(sys.argv) > 1 else 0
    try:
        cam_source = int(cam_source)
    except ValueError:
        pass

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=0
    )

    print(f"[INFO] Membuka kamera: {cam_source} ...")
    cap = cv2.VideoCapture(cam_source)
    if isinstance(cam_source, str):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam /dev/video0.")
        return

    # Inisialisasi CSV jika belum ada
    if not os.path.exists(CSV_FILE):
        header = ["label"] + [f"f_{i}" for i in range(42)]
        with open(CSV_FILE, "w") as f:
            f.write(",".join(header) + "\n")

    # Hitung jumlah sampel yang sudah ada
    counts = {k: 0 for k in CLASS_MAP.keys()}
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        try:
            df_existing = pd.read_csv(CSV_FILE)
            for k in counts:
                counts[k] = int((df_existing["label"] == k).sum())
        except Exception:
            pass

    print("==================================================")
    print("Hand Gesture Data Collector")
    print("Tahan pose tangan, lalu TEKAN DAN TAHAN tombol angka:")
    print(" [0] : STANDBY (Telapak Tangan Terbuka)")
    print(" [1] : MOVE (Telunjuk)")
    print(" [2] : SCROLL (V / Telunjuk + Jari Tengah)")
    print(" [3] : LEFT_CLICK (3 jari terbuka)")
    print(" [4] : RIGHT_CLICK (4 jari terbuka)")
    print(" [q] : Selesai merekam")
    print("==================================================")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        features = None
        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            features = extract_features(hand_lm.landmark)

        # Panel Informasi di Layar
        cv2.rectangle(frame, (10, 10), (320, 160), (30, 30, 30), -1)
        cv2.rectangle(frame, (10, 10), (320, 160), (70, 70, 70), 1)

        y_offset = 32
        for k, name in CLASS_MAP.items():
            text = f"Key [{k}] {name:<11}: {counts[k]} data"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            y_offset += 22

        cv2.putText(frame, "Tekan 'q' untuk selesai", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow("Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key in [ord(str(i)) for i in CLASS_MAP.keys()]:
            label = int(chr(key))
            if features is not None:
                row = [str(label)] + [f"{v:.6f}" for v in features]
                with open(CSV_FILE, "a") as f:
                    f.write(",".join(row) + "\n")
                counts[label] += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nPengumpulan data selesai! Total data tersimpan di {CSV_FILE}:")
    for k, name in CLASS_MAP.items():
        print(f" - {name} ({k}): {counts[k]} sampel")

if __name__ == "__main__":
    main()
