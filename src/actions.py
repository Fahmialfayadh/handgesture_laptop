"""
OS input injection module for mouse cursor, clicks, and scrolling.
"""

import math
import time
from pynput.mouse import Button
from src.utils import map_to_screen

class ActionEngine:
    """
    Eksekusi aksi sistem operasi (kursor, klik, scroll) berdasarkan mode gestur yang aktif.
    """

    def __init__(self, mouse, screen_width, screen_height):
        self.mouse = mouse
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cursor_x = screen_width / 2
        self.cursor_y = screen_height / 2
        self.last_click = 0.0
        self.click_fired = False
        self.scroll_anchor_y = None
        self.last_scroll_time = 0.0
        self.current_mode = None
        self.handlers = {
            "MOVE": self.move,
            "SCROLL": self.scroll,
            "LEFT_CLICK": self.left_click,
            "RIGHT_CLICK": self.right_click,
            "STANDBY": self.idle,
            "NO HAND": self.idle,
        }

    def reset_motion(self):
        """Reset anchor pergerakan saat berganti mode gestur."""
        self.scroll_anchor_y = None
        self.last_scroll_time = 0.0
        self.click_fired = False

    def dispatch(self, mode, landmarks, frame_width, frame_height):
        """Arahkan eksekusi ke handler yang sesuai dengan mode gestur."""
        if mode != self.current_mode:
            self.reset_motion()
            self.current_mode = mode
        self.handlers.get(mode, self.idle)(landmarks, frame_width, frame_height)

    def move(self, landmarks, frame_width, frame_height):
        """Gerakkan kursor dengan peredaman adaptif (presisi saat pelan, gesit saat cepat)."""
        target_x, target_y = map_to_screen(
            landmarks[8], frame_width, frame_height,
            self.screen_width, self.screen_height,
        )
        distance = math.hypot(target_x - self.cursor_x, target_y - self.cursor_y)
        if distance < 2.5:
            return

        alpha = 0.12 + min(0.55, distance / 650.0)
        self.cursor_x += alpha * (target_x - self.cursor_x)
        self.cursor_y += alpha * (target_y - self.cursor_y)
        self.mouse.position = (int(self.cursor_x), int(self.cursor_y))

    def scroll(self, landmarks, frame_width, frame_height):
        """Scroll halaman kontinu berbasis anchor titik awal dengan zona netral."""
        raw_y = landmarks[9].y * frame_height
        if self.scroll_anchor_y is None:
            self.scroll_anchor_y = raw_y
            self.last_scroll_time = time.monotonic()
            return

        deadzone_px = 8.0
        offset = self.scroll_anchor_y - raw_y
        if abs(offset) < deadzone_px:
            return

        now = time.monotonic()
        scroll_interval = 0.075
        if now - self.last_scroll_time < scroll_interval:
            return

        direction = 1 if offset > 0 else -1
        steps = direction * min(3, max(1, int(abs(offset) / deadzone_px)))
        self.mouse.scroll(0, steps)
        self.last_scroll_time = now

    def left_click(self, landmarks, frame_width, frame_height):
        """Klik kiri tunggal dengan proteksi debounce."""
        if not self.click_fired and time.time() - self.last_click >= 0.55:
            self.mouse.click(Button.left)
            self.last_click = time.time()
            self.click_fired = True

    def right_click(self, landmarks, frame_width, frame_height):
        """Klik kanan tunggal dengan proteksi debounce."""
        if not self.click_fired and time.time() - self.last_click >= 0.65:
            self.mouse.click(Button.right)
            self.last_click = time.time()
            self.click_fired = True

    def idle(self, landmarks, frame_width, frame_height):
        """Mode diam / netral."""
        return

