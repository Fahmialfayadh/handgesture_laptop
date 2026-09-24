"""
Utility functions for feature extraction and screen coordinate mapping.
"""

import math

def extract_features(landmarks):
    """
    Mengekstrak 42 koordinat normalisasi sendi tangan terhadap pergelangan tangan (wrist).
    Input: 21 landmark dari MediaPipe.
    Output: list 42 float yang diskalakan terhadap panjang telapak tangan.
    """
    base_x, base_y = landmarks[0].x, landmarks[0].y
    raw = [coord for pt in landmarks for coord in (pt.x - base_x, pt.y - base_y)]
    scale = math.hypot(landmarks[9].x - base_x, landmarks[9].y - base_y) or 1.0
    return [v / scale for v in raw]

def map_to_screen(tip_lm, frame_w, frame_h, screen_w, screen_h, margin_x=100, margin_y=80):
    """
    Memetakan posisi ujung telunjuk (tip) ke koordinat layar laptop dengan margin kalibrasi.
    """
    clamped_x = max(margin_x, min(frame_w - margin_x, tip_lm.x * frame_w))
    cy = max(margin_y, min(frame_h - margin_y, tip_lm.y * frame_h))
    norm_x = (clamped_x - margin_x) / (frame_w - 2 * margin_x)
    norm_y = (cy - margin_y) / (frame_h - 2 * margin_y)
    return norm_x * screen_w, norm_y * screen_h

