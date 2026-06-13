"""
Filename : RTOD_v3.py
Version  : 3 - Realtime Webcam Object Measurement
Usage    :
    Jalankan -> pilih kamera dari dropdown -> klik Connect.
    Referensi: Lingkaran diameter 3 cm + Persegi 3x3 cm (sisi kiri frame).
    Tekan S untuk screenshot.  Tekan R untuk reset kalibrasi.

Changelog v2 -> v3:
    [FIX] Camera scan di QThread - tidak ada UI freeze saat startup
    [FIX] QImage buffer safety - bytes() agar tidak segfault
    [FIX] Keyboard shortcut S (screenshot) dan R (reset) berfungsi
    [FIX] Cross-platform camera backend (CAP_DSHOW hanya di Windows)
    [FIX] statusBar di-style sekali di __init__
    [FIX] MeasRow update in-place, tidak destroy/recreate tiap frame
    [FIX] CPU throttle di CameraThread
    [FIX] is_circle dan is_square dicek independen (bukan elif)
    [FIX] Semua karakter Unicode/emoji diganti ASCII - tidak ada tampilan ???
    [IMPROVE] Adaptive Canny berbasis Otsu
    [IMPROVE] PPCStabilizer - kunci px/cm setelah stabil
    [IMPROVE] DimensionEMA - smoothing output dimensi
    [IMPROVE] Solidity check pada is_circle
    [IMPROVE] MIN_REF_AREA - filter noise jadi referensi palsu
    [IMPROVE] Bounding circle untuk objek bulat, label D (diameter)
    [IMPROVE] UI research style - bersih, monospace, tanpa dekorasi
"""

import sys
import os
import time
import cv2
import numpy as np
import imutils
from imutils import perspective, contours
from scipy.spatial.distance import euclidean
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy,
    QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette,
    QBrush, QPainter
)

current_folder = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_DIR = os.path.join(current_folder, "screenshots")

if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

# --------------------------------------------------
#  KONFIGURASI
# --------------------------------------------------
REF_CIRCLE_DIAM  = 3.0    # diameter lingkaran referensi (cm)
REF_SQUARE_SIDE  = 3.0    # sisi persegi referensi (cm)
MAX_CAMERAS      = 6

# Deteksi bentuk
CIRCLE_THRESH    = 0.82   # circularity minimum (0-1)
CIRCLE_SOLIDITY  = 0.90   # solidity minimum untuk lingkaran
SQUARE_TOL       = 0.15   # toleransi rasio w/h persegi

# Filter kontour
MIN_CONTOUR_AREA = 100    # area minimum semua kontour (px^2)
MIN_REF_AREA     = 1500   # area minimum kontour kandidat referensi (px^2)

# Stabilisasi PPC
PPC_LOCK_FRAMES  = 20     # jumlah frame sebelum kunci ppc
PPC_LOCK_TOL     = 0.04   # toleransi std/mean (4%)

# Smoothing dimensi
EMA_ALPHA        = 0.20   # alpha EMA

# Target FPS
TARGET_FPS       = 30

# --------------------------------------------------
C_BG        = "#F4F4F4"
C_SURFACE   = "#FFFFFF"
C_SURFACE2  = "#EBEBEB"
C_BORDER    = "#C8C8C8"
C_HEADER    = "#1A1A2E"
C_ACCENT    = "#003580"
C_GREEN     = "#1B6B3A"
C_RED       = "#B00020"
C_ORANGE    = "#7A4000"
C_TEXT      = "#111111"
C_TEXT_DIM  = "#666666"
C_TEXT_INV  = "#FFFFFF"
C_VIDEO_BG  = "#111111"


# --------------------------------------------------
#  STABILIZER
# --------------------------------------------------

class PPCStabilizer:
    """
    Kumpulkan ppc selama N frame.
    Kunci nilai jika variasi di bawah toleransi.
    Auto-reset jika nilai tiba-tiba bergeser lebih dari 15%.
    """
    def __init__(self, n=PPC_LOCK_FRAMES, tolerance=PPC_LOCK_TOL):
        self.buffer     = deque(maxlen=n)
        self.n          = n
        self.tol        = tolerance
        self.locked_ppc = None
        self.is_locked  = False

    def update(self, ppc):
        if ppc is None:
            return self.locked_ppc, self.is_locked

        self.buffer.append(ppc)

        if self.is_locked:
            if abs(ppc - self.locked_ppc) / self.locked_ppc > 0.15:
                self.reset()
            return self.locked_ppc, self.is_locked

        if len(self.buffer) == self.n:
            arr  = np.array(self.buffer)
            mean = arr.mean()
            std  = arr.std()
            if mean > 0 and std / mean < self.tol:
                self.locked_ppc = float(mean)
                self.is_locked  = True

        current = float(np.mean(self.buffer)) if self.buffer else None
        return current, self.is_locked

    def reset(self):
        self.buffer.clear()
        self.locked_ppc = None
        self.is_locked  = False


class DimensionEMA:
    """Exponential Moving Average dimensi per indeks objek."""
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha  = alpha
        self.values = {}

    def update(self, idx, w, h):
        if idx not in self.values:
            self.values[idx] = (w, h)
        else:
            ew, eh = self.values[idx]
            self.values[idx] = (
                self.alpha * w + (1.0 - self.alpha) * ew,
                self.alpha * h + (1.0 - self.alpha) * eh,
            )
        return self.values[idx]

    def reset(self):
        self.values.clear()


# --------------------------------------------------
#  COMPUTER VISION HELPERS
# --------------------------------------------------

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def adaptive_canny(blur):
    """Threshold Canny otomatis berbasis Otsu."""
    otsu_thresh, _ = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    low  = max(10.0, 0.5 * otsu_thresh)
    high = max(30.0, otsu_thresh)
    return cv2.Canny(blur, low, high)


def is_circle(cnt, thresh=CIRCLE_THRESH, min_solidity=CIRCLE_SOLIDITY,
              min_area=MIN_REF_AREA):
    """
    Klasifikasi lingkaran: circularity + solidity.
    min_area: MIN_REF_AREA untuk referensi, MIN_CONTOUR_AREA untuk objek biasa.
    """
    area = cv2.contourArea(cnt)
    if area < min_area:
        return False
    perim = cv2.arcLength(cnt, True)
    if perim == 0:
        return False
    circularity = (4.0 * np.pi * area) / (perim ** 2)
    if circularity < thresh:
        return False
    hull      = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False
    return (area / hull_area) >= min_solidity


def is_square(cnt, tol=SQUARE_TOL):
    """Klasifikasi persegi: 4 sudut + rasio w/h + solidity."""
    if cv2.contourArea(cnt) < MIN_REF_AREA:
        return False
    peri   = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) != 4:
        return False
    x, y, w, h = cv2.boundingRect(approx)
    if h == 0:
        return False
    if abs(1.0 - w / float(h)) > tol:
        return False
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area == 0:
        return False
    return (cv2.contourArea(cnt) / hull_area) > 0.85


def compute_scale(cnts_sorted):
    """
    Cari lingkaran dan persegi secara independen (dua pass terpisah).
    Return: (ppc, ref_idx_list, found_dict)
    """
    circle_idx = square_idx = None

    for i, c in enumerate(cnts_sorted):
        if is_circle(c):
            circle_idx = i
            break

    for i, c in enumerate(cnts_sorted):
        if i == circle_idx:
            continue
        if is_square(c):
            square_idx = i
            break

    ratios = []
    found  = {
        "circle": circle_idx is not None,
        "square": square_idx is not None,
    }

    if circle_idx is not None:
        _, radius = cv2.minEnclosingCircle(cnts_sorted[circle_idx])
        ratios.append((2.0 * radius) / REF_CIRCLE_DIAM)

    if square_idx is not None:
        box = cv2.minAreaRect(cnts_sorted[square_idx])
        pts = perspective.order_points(
            np.array(cv2.boxPoints(box), dtype="float32")
        )
        tl, tr, br, _ = pts
        side = (euclidean(tl, tr) + euclidean(tr, br)) / 2.0
        ratios.append(side / REF_SQUARE_SIDE)

    ref_idx = [i for i in (circle_idx, square_idx) if i is not None]
    ppc     = float(np.mean(ratios)) if ratios else None
    return ppc, ref_idx, found


def draw_measurement(frame, cnt, ppc, is_ref=False, shape="rect"):
    """
    Gambar bounding sesuai shape:
      shape="circle" -> lingkaran + garis silang + label D
      shape="rect"   -> bounding rect + label W / H
    Return:
      (diameter, None) untuk circle
      (width, height)  untuk rect
    """
    color_ref = (0,  80, 180)
    color_obj = (0, 140,  70)
    color     = color_ref if is_ref else color_obj
    txt_color = (255, 255,  50)

    if shape == "circle":
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cxi = int(cx)
        cyi = int(cy)
        ri  = int(radius)
        diam = (2.0 * radius) / ppc

        cv2.circle(frame, (cxi, cyi), ri, color, 2)
        cv2.circle(frame, (cxi, cyi), 3, color, -1)
        cv2.line(frame, (cxi - ri, cyi), (cxi + ri, cyi), color, 1)
        cv2.line(frame, (cxi, cyi - ri), (cxi, cyi + ri), color, 1)

        label  = f"D: {diam:.1f} cm"
        text_x = max(cxi - ri, 2)
        text_y = max(cyi - ri - 6, 14)
        cv2.putText(frame, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_color, 2)
        return diam, None

    else:
        box = cv2.minAreaRect(cnt)
        pts = perspective.order_points(
            np.array(cv2.boxPoints(box), dtype="float32")
        )
        tl, tr, br, bl = pts
        cv2.drawContours(frame, [pts.astype("int")], -1, color, 2)

        mid_t    = midpoint(tl, tr)
        mid_r    = midpoint(tr, br)
        wid      = euclidean(tl, tr) / ppc
        ht       = euclidean(tr, br) / ppc
        box_h    = euclidean(tl, bl)
        offset_y = max(12, int(box_h * 0.07))

        cv2.putText(frame, f"W:{wid:.1f}cm",
                    (int(mid_t[0] - 22), int(mid_t[1] - offset_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_color, 2)
        cv2.putText(frame, f"H:{ht:.1f}cm",
                    (int(mid_r[0] + 8), int(mid_r[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_color, 2)
        return wid, ht


def process_frame(frame, ppc_override=None):
    """
    Proses satu frame: Adaptive Canny, kontour, skala, anotasi.
    Return: (annotated_frame, raw_ppc, found_dict, measurements_list)
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (9, 9), 0)
    edged = adaptive_canny(blur)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    raw = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    raw = imutils.grab_contours(raw)
    if not raw:
        return frame, None, {"circle": False, "square": False}, []

    (cnts_s, _) = contours.sort_contours(raw)
    cnts_s = [c for c in cnts_s if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    if not cnts_s:
        return frame, None, {"circle": False, "square": False}, []

    raw_ppc, ref_idx, found = compute_scale(cnts_s)
    ppc = ppc_override if ppc_override is not None else raw_ppc

    measurements = []
    if ppc:
        for i, cnt in enumerate(cnts_s):
            is_r  = (i in ref_idx)
            shape = (
                "circle"
                if is_circle(cnt, min_area=MIN_CONTOUR_AREA)
                else "rect"
            )
            val_a, val_b = draw_measurement(
                frame, cnt, ppc, is_ref=is_r, shape=shape
            )
            measurements.append({
                "index":  i + 1,
                "shape":  shape,
                "w":      val_a,
                "h":      val_b,
                "is_ref": is_r,
            })

    return frame, raw_ppc, found, measurements


# --------------------------------------------------
#  CAMERA SCAN THREAD
# --------------------------------------------------

class CameraScanThread(QThread):
    """Scan kamera di background agar UI tidak freeze."""
    scan_done = pyqtSignal(list)

    def run(self):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        found   = []
        for i in range(MAX_CAMERAS):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    found.append(i)
            cap.release()
        self.scan_done.emit(found)


# --------------------------------------------------
#  CAMERA THREAD
# --------------------------------------------------

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, object, dict, list)
    error       = pyqtSignal(str)

    def __init__(self, cam_idx):
        super().__init__()
        self.cam_idx         = cam_idx
        self._running        = True
        self.ppc_stabilizer  = PPCStabilizer()
        self.ema             = DimensionEMA()
        self._frame_interval = 1.0 / TARGET_FPS

    def run(self):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cam_idx, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {self.cam_idx}")
            return

        while self._running:
            t0 = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                self.error.emit("Frame read failed.")
                break

            ppc_locked = (
                self.ppc_stabilizer.locked_ppc
                if self.ppc_stabilizer.is_locked else None
            )

            annotated, raw_ppc, found, measurements = process_frame(
                frame.copy(), ppc_override=ppc_locked
            )

            stable_ppc, _ = self.ppc_stabilizer.update(raw_ppc)

            # EMA smoothing
            smoothed = []
            for m in measurements:
                if m["h"] is None:
                    # circle: smooth diameter saja
                    d = m["w"]
                    sd, _ = self.ema.update(m["index"], d, d)
                    smoothed.append({**m, "w": sd, "h": None})
                else:
                    sw, sh = self.ema.update(m["index"], m["w"], m["h"])
                    smoothed.append({**m, "w": sw, "h": sh})

            self.frame_ready.emit(annotated, stable_ppc, found, smoothed)

            elapsed = time.monotonic() - t0
            sleep_t = self._frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        cap.release()

    def stop(self):
        self._running = False
        self.wait()

    def reset_stabilizer(self):
        self.ppc_stabilizer.reset()
        self.ema.reset()


# --------------------------------------------------
#  UI KOMPONEN KUSTOM
# --------------------------------------------------

class StatusDot(QWidget):
    """Indikator bulat kecil merah/hijau."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self._color = QColor(C_RED)

    def set_ok(self, ok: bool):
        self._color = QColor(C_GREEN) if ok else QColor(C_RED)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self._color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(1, 1, 10, 10)


class Divider(QFrame):
    """Garis pemisah horizontal tipis."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Plain)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {C_BORDER}; border: none;")


class MeasRow(QWidget):
    """
    Satu baris hasil pengukuran.
    shape='circle': tampilkan D (diameter)
    shape='rect':   tampilkan W dan H
    Mendukung update nilai in-place via update_values().
    """
    def __init__(self, index, w, h, is_ref, shape="rect", parent=None):
        super().__init__(parent)
        self._shape = shape

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(0)

        # Kolom ID
        tag       = "REF" if is_ref else str(index).zfill(2)
        tag_color = C_ORANGE if is_ref else C_ACCENT

        self.tag_lbl = QLabel(tag)
        self.tag_lbl.setFixedWidth(32)
        self.tag_lbl.setAlignment(Qt.AlignCenter)
        self.tag_lbl.setStyleSheet(f"""
            color: {tag_color};
            font-size: 11px;
            font-weight: bold;
            font-family: Consolas, "Courier New", monospace;
            padding-right: 4px;
            border-right: 1px solid {C_BORDER};
        """)

        # Kolom Shape
        shape_txt = "Circle" if shape == "circle" else "Rect  "
        self.shape_lbl = QLabel(shape_txt)
        self.shape_lbl.setFixedWidth(44)
        self.shape_lbl.setStyleSheet(f"""
            color: {C_TEXT_DIM};
            font-size: 11px;
            font-family: Consolas, "Courier New", monospace;
            padding-left: 6px;
        """)

        # Kolom dimensi
        if shape == "circle":
            dim_text = f"D: {w:.2f} cm"
        else:
            dim_text = f"W: {w:.2f}  H: {h:.2f} cm"

        self.dim_lbl = QLabel(dim_text)
        self.dim_lbl.setStyleSheet(f"""
            color: {C_TEXT};
            font-size: 12px;
            font-family: Consolas, "Courier New", monospace;
        """)

        layout.addWidget(self.tag_lbl)
        layout.addWidget(self.shape_lbl)
        layout.addWidget(self.dim_lbl)
        layout.addStretch()

        bg = "#FFF8F0" if is_ref else C_SURFACE
        self.setStyleSheet(f"""
            QWidget {{
                background: {bg};
                border-bottom: 1px solid {C_BORDER};
            }}
        """)
        self.setFixedHeight(28)

    def update_values(self, w, h):
        """Update teks in-place tanpa rebuild widget."""
        if self._shape == "circle":
            self.dim_lbl.setText(f"D: {w:.2f} cm")
        else:
            if h is not None:
                self.dim_lbl.setText(f"W: {w:.2f}  H: {h:.2f} cm")


# --------------------------------------------------
#  MAIN WINDOW
# --------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Measurement System  v3")
        self.setMinimumSize(1100, 700)

        self.camera_thread   = None
        self.last_frame      = None
        self._scan_thread    = None
        self._frame_count    = 0
        self._meas_rows      = []
        self._screenshot_dir = SCREENSHOT_DIR

        self._setup_palette()
        self._build_ui()
        self._style_statusbar()
        self._scan_cameras()

    # -- Palette ----------------------------------------------
    def _setup_palette(self):
        pal = QPalette()
        pal.setColor(QPalette.Window,     QColor(C_BG))
        pal.setColor(QPalette.WindowText, QColor(C_TEXT))
        pal.setColor(QPalette.Base,       QColor(C_SURFACE))
        pal.setColor(QPalette.Text,       QColor(C_TEXT))
        self.setPalette(pal)
        self.setStyleSheet(f"QMainWindow {{ background: {C_BG}; }}")

    def _style_statusbar(self):
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background: {C_SURFACE};
                color: {C_TEXT_DIM};
                border-top: 1px solid {C_BORDER};
                font-size: 11px;
                font-family: Consolas, "Courier New", monospace;
                padding: 2px 8px;
            }}
        """)

    # -- Build UI ----------------------------------------------
    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet(f"background: {C_BG};")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # -- Header bar ---------------------------------------
        header = QWidget()
        header.setFixedHeight(44)
        header.setStyleSheet(f"background: {C_HEADER};")

        hdr_lay = QHBoxLayout(header)
        hdr_lay.setContentsMargins(16, 0, 16, 0)

        title_lbl = QLabel("OBJECT MEASUREMENT SYSTEM  v3")
        title_lbl.setStyleSheet(f"""
            color: {C_TEXT_INV};
            font-size: 14px;
            font-weight: bold;
            font-family: Consolas, "Courier New", monospace;
            letter-spacing: 2px;
        """)

        self.fps_lbl = QLabel("FPS: --")
        self.fps_lbl.setStyleSheet(f"""
            color: {C_TEXT_INV};
            font-size: 11px;
            font-family: Consolas, "Courier New", monospace;
        """)

        hdr_lay.addWidget(title_lbl)
        hdr_lay.addStretch()
        hdr_lay.addWidget(self.fps_lbl)
        outer.addWidget(header)

        # -- Body ---------------------------------------------
        body = QWidget()
        body.setStyleSheet(f"background: {C_BG};")
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(12, 12, 12, 12)
        body_lay.setSpacing(12)

        # -- LEFT: video + info bar --
        left_v = QVBoxLayout()
        left_v.setSpacing(6)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(720, 500)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.video_label.setStyleSheet(f"""
            background: {C_VIDEO_BG};
            border: 1px solid {C_BORDER};
            color: {C_TEXT_DIM};
            font-size: 13px;
            font-family: Consolas, "Courier New", monospace;
        """)
        self.video_label.setText(
            "No camera feed\n\nSelect a camera and click Connect"
        )
        left_v.addWidget(self.video_label, 1)

        # Info bar bawah video
        info_bar = QHBoxLayout()
        info_bar.setSpacing(20)

        self.scale_lbl = QLabel("Scale: --")
        self.scale_lbl.setStyleSheet(f"""
            color: {C_TEXT};
            font-size: 11px;
            font-family: Consolas, "Courier New", monospace;
        """)

        self.lock_lbl = QLabel("Calibration: UNLOCKED")
        self.lock_lbl.setStyleSheet(f"""
            color: {C_RED};
            font-size: 11px;
            font-weight: bold;
            font-family: Consolas, "Courier New", monospace;
        """)

        info_bar.addWidget(self.scale_lbl)
        info_bar.addWidget(self.lock_lbl)
        info_bar.addStretch()
        left_v.addLayout(info_bar)
        body_lay.addLayout(left_v, 1)

        # -- RIGHT: panel kontrol --
        panel = QFrame()
        panel.setFixedWidth(276)
        panel.setStyleSheet(f"""
            QFrame {{
                background: {C_SURFACE};
                border: 1px solid {C_BORDER};
            }}
        """)
        panel_v = QVBoxLayout(panel)
        panel_v.setContentsMargins(0, 0, 0, 0)
        panel_v.setSpacing(0)

        # Bagian: Camera
        self._panel_section(panel_v, "CAMERA")
        cam_body = self._panel_body()

        self.cam_combo = QComboBox()
        self.cam_combo.addItem("Scanning...")
        self.cam_combo.setEnabled(False)
        self.cam_combo.setStyleSheet(f"""
            QComboBox {{
                background: {C_SURFACE};
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                padding: 5px 8px;
                font-size: 12px;
                font-family: Consolas, "Courier New", monospace;
            }}
            QComboBox::drop-down {{ border: none; width: 18px; }}
            QComboBox QAbstractItemView {{
                background: {C_SURFACE};
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                selection-background-color: {C_ACCENT};
                selection-color: {C_TEXT_INV};
                font-family: Consolas, "Courier New", monospace;
            }}
        """)
        cam_body.addWidget(self.cam_combo)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setEnabled(False)
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_connect.setStyleSheet(self._btn_style(C_ACCENT))

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setStyleSheet(self._btn_style(C_RED))

        btn_row.addWidget(self.btn_connect)
        btn_row.addWidget(self.btn_stop)
        cam_body.addLayout(btn_row)

        self.btn_reset = QPushButton("Reset Calibration  [R]")
        self.btn_reset.setEnabled(False)
        self.btn_reset.clicked.connect(self._on_reset_stabilizer)
        self.btn_reset.setStyleSheet(self._btn_style(C_TEXT_DIM))
        cam_body.addWidget(self.btn_reset)

        cam_widget = QWidget()
        cam_widget.setStyleSheet(f"background: {C_SURFACE};")
        cam_widget.setLayout(cam_body)
        panel_v.addWidget(cam_widget)
        panel_v.addWidget(Divider())

        # Bagian: Reference Objects
        self._panel_section(panel_v, "REFERENCE OBJECTS")
        ref_body = self._panel_body()
        ref_body.setSpacing(8)

        row_c = QHBoxLayout()
        self.dot_circle = StatusDot()
        circ_lbl = QLabel(
            f"Circle   D = {REF_CIRCLE_DIAM:.1f} cm"
        )
        circ_lbl.setStyleSheet(
            f"color: {C_TEXT}; font-size: 12px; "
            f"font-family: Consolas, 'Courier New', monospace;"
        )
        row_c.addWidget(self.dot_circle)
        row_c.addSpacing(8)
        row_c.addWidget(circ_lbl)
        row_c.addStretch()
        ref_body.addLayout(row_c)

        row_s = QHBoxLayout()
        self.dot_square = StatusDot()
        sq_lbl = QLabel(
            f"Square   {REF_SQUARE_SIDE:.1f} x {REF_SQUARE_SIDE:.1f} cm"
        )
        sq_lbl.setStyleSheet(
            f"color: {C_TEXT}; font-size: 12px; "
            f"font-family: Consolas, 'Courier New', monospace;"
        )
        row_s.addWidget(self.dot_square)
        row_s.addSpacing(8)
        row_s.addWidget(sq_lbl)
        row_s.addStretch()
        ref_body.addLayout(row_s)

        hint = QLabel("Place references on the left side of frame")
        hint.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 10px; font-style: italic;"
        )
        ref_body.addWidget(hint)

        ref_widget = QWidget()
        ref_widget.setStyleSheet(f"background: {C_SURFACE};")
        ref_widget.setLayout(ref_body)
        panel_v.addWidget(ref_widget)
        panel_v.addWidget(Divider())

        # Bagian: Measurements - header tabel
        self._panel_section(panel_v, "MEASUREMENTS")

        tbl_hdr = QWidget()
        tbl_hdr.setFixedHeight(24)
        tbl_hdr.setStyleSheet(f"background: {C_SURFACE2}; border-bottom: 1px solid {C_BORDER};")
        tbl_h_lay = QHBoxLayout(tbl_hdr)
        tbl_h_lay.setContentsMargins(8, 0, 8, 0)
        tbl_h_lay.setSpacing(0)

        for col_text, col_width in [("ID", 32), ("TYPE", 44), ("DIMENSION", 0)]:
            lbl = QLabel(col_text)
            lbl.setStyleSheet(f"""
                color: {C_TEXT_DIM};
                font-size: 9px;
                font-weight: bold;
                font-family: Consolas, "Courier New", monospace;
                letter-spacing: 1px;
            """)
            if col_width > 0:
                lbl.setFixedWidth(col_width)
            tbl_h_lay.addWidget(lbl)

        tbl_h_lay.addStretch()
        panel_v.addWidget(tbl_hdr)

        # Scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(f"""
            QScrollArea {{ border: none; background: {C_SURFACE}; }}
            QScrollBar:vertical {{
                background: {C_SURFACE2};
                width: 4px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {C_BORDER};
                border-radius: 2px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)

        self.meas_container = QWidget()
        self.meas_container.setStyleSheet(f"background: {C_SURFACE};")
        self.meas_layout = QVBoxLayout(self.meas_container)
        self.meas_layout.setSpacing(0)
        self.meas_layout.setContentsMargins(0, 0, 0, 0)
        self.meas_layout.addStretch()

        self.scroll.setWidget(self.meas_container)
        panel_v.addWidget(self.scroll, 1)
        panel_v.addWidget(Divider())

        # Screenshot button
        self.btn_screenshot = QPushButton("Save Screenshot  [S]")
        self.btn_screenshot.setEnabled(False)
        self.btn_screenshot.clicked.connect(self._on_screenshot)
        self.btn_screenshot.setStyleSheet(
            self._btn_style(C_GREEN, padding="8px 14px", margin="8px")
        )
        panel_v.addWidget(self.btn_screenshot)

        body_lay.addWidget(panel)
        outer.addWidget(body, 1)

        # FPS timer
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._reset_fps)
        self._fps_timer.start(1000)

    # -- Panel helpers -----------------------------------------
    def _panel_section(self, parent_layout, text):
        """Header section tipis di dalam panel."""
        w = QWidget()
        w.setFixedHeight(26)
        w.setStyleSheet(f"background: {C_SURFACE2};")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(10, 0, 10, 0)
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            color: {C_TEXT_DIM};
            font-size: 9px;
            font-weight: bold;
            font-family: Consolas, "Courier New", monospace;
            letter-spacing: 1px;
        """)
        lay.addWidget(lbl)
        lay.addStretch()
        parent_layout.addWidget(w)
        parent_layout.addWidget(Divider())

    def _panel_body(self):
        """Buat QVBoxLayout dengan margin standar untuk isi section."""
        lay = QVBoxLayout()
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        return lay

    # -- Button style -----------------------------------------
    def _btn_style(self, color, padding="6px 12px", margin="0px"):
        return f"""
            QPushButton {{
                background: {color};
                color: {C_TEXT_INV};
                border: none;
                padding: {padding};
                margin: {margin};
                font-size: 11px;
                font-family: Consolas, "Courier New", monospace;
            }}
            QPushButton:hover {{
                background: {color}CC;
            }}
            QPushButton:disabled {{
                background: {C_BORDER};
                color: {C_TEXT_DIM};
            }}
            QPushButton:pressed {{
                background: {color}AA;
            }}
        """

    # -- Camera scan ------------------------------------------
    def _scan_cameras(self):
        self.cam_combo.clear()
        self.cam_combo.addItem("Scanning cameras...")
        self.cam_combo.setEnabled(False)
        self.btn_connect.setEnabled(False)

        self._scan_thread = CameraScanThread()
        self._scan_thread.scan_done.connect(self._on_scan_done)
        self._scan_thread.start()

    def _on_scan_done(self, found):
        self.cam_combo.clear()
        if found:
            for idx in found:
                suffix = " (built-in)" if idx == 0 else ""
                self.cam_combo.addItem(f"Camera {idx}{suffix}", userData=idx)
            self.cam_combo.setEnabled(True)
            self.btn_connect.setEnabled(True)
            self.statusBar().showMessage(
                f"{len(found)} camera(s) found.", 3000
            )
        else:
            self.cam_combo.addItem("No camera found")
            self.statusBar().showMessage("No camera detected.", 3000)

    # -- Slots ------------------------------------------------
    def _on_connect(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

        cam_idx = self.cam_combo.currentData()
        if cam_idx is None:
            return

        self.camera_thread = CameraThread(cam_idx)
        self.camera_thread.frame_ready.connect(self._on_frame)
        self.camera_thread.error.connect(self._on_cam_error)
        self.camera_thread.start()

        self.btn_connect.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_screenshot.setEnabled(True)
        self.video_label.setText("")

    def _on_stop(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

        self.btn_connect.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.btn_screenshot.setEnabled(False)
        self.video_label.setText(
            "No camera feed\n\nSelect a camera and click Connect"
        )
        self._update_ref_dots({"circle": False, "square": False})
        self.scale_lbl.setText("Scale: --")
        self._set_lock_label(False)
        self._clear_measurements()

    def _on_frame(self, frame, ppc, found, measurements):
        self._frame_count += 1
        self.last_frame = frame

        # Render frame
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch  = rgb.shape
        img_bytes = bytes(rgb.data)
        qi  = QImage(img_bytes, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qi).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

        # Update status
        self._update_ref_dots(found)

        if ppc:
            self.scale_lbl.setText(f"Scale: {ppc:.3f} px/cm")
        else:
            self.scale_lbl.setText("Scale: --")

        if self.camera_thread:
            self._set_lock_label(self.camera_thread.ppc_stabilizer.is_locked)

        if self._frame_count % 8 == 0:
            self._refresh_measurements(measurements)

    def _on_cam_error(self, msg):
        self.statusBar().showMessage(f"Camera error: {msg}", 5000)
        self._on_stop()

    def _on_screenshot(self):
        if self.last_frame is None:
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"measurement_{ts}.jpg"
        path = os.path.join(self._screenshot_dir, name)
        cv2.imwrite(path, self.last_frame)
        self.statusBar().showMessage(f"Saved: {path}", 4000)

    def _on_reset_stabilizer(self):
        if self.camera_thread:
            self.camera_thread.reset_stabilizer()
        self._set_lock_label(False)
        self.scale_lbl.setText("Scale: --")
        self.statusBar().showMessage(
            "Calibration reset. Waiting for stable reference...", 3000
        )

    # -- UI helpers -------------------------------------------
    def _update_ref_dots(self, found):
        self.dot_circle.set_ok(found.get("circle", False))
        self.dot_square.set_ok(found.get("square", False))

    def _set_lock_label(self, is_locked: bool):
        if is_locked:
            self.lock_lbl.setText("Calibration: LOCKED")
            self.lock_lbl.setStyleSheet(f"""
                color: {C_GREEN};
                font-size: 11px;
                font-weight: bold;
                font-family: Consolas, "Courier New", monospace;
            """)
        else:
            self.lock_lbl.setText("Calibration: UNLOCKED")
            self.lock_lbl.setStyleSheet(f"""
                color: {C_RED};
                font-size: 11px;
                font-weight: bold;
                font-family: Consolas, "Courier New", monospace;
            """)

    def _clear_measurements(self):
        for row in self._meas_rows:
            row.deleteLater()
        self._meas_rows.clear()

    def _refresh_measurements(self, measurements):
        new_count = len(measurements)
        old_count = len(self._meas_rows)

        shapes_match = (
            new_count == old_count and
            all(
                r._shape == m["shape"]
                for r, m in zip(self._meas_rows, measurements)
            )
        )

        if shapes_match:
            for row, m in zip(self._meas_rows, measurements):
                row.update_values(m["w"], m["h"])
            return

        self._clear_measurements()
        for m in measurements:
            row = MeasRow(
                m["index"], m["w"], m["h"],
                m["is_ref"], shape=m["shape"]
            )
            self.meas_layout.insertWidget(self.meas_layout.count() - 1, row)
            self._meas_rows.append(row)

    def _reset_fps(self):
        self.fps_lbl.setText(f"FPS: {self._frame_count}")
        self._frame_count = 0

    # -- Keyboard shortcuts -----------------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S and self.btn_screenshot.isEnabled():
            self._on_screenshot()
        elif event.key() == Qt.Key_R and self.btn_reset.isEnabled():
            self._on_reset_stabilizer()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        if self._scan_thread and self._scan_thread.isRunning():
            self._scan_thread.wait()
        event.accept()


# --------------------------------------------------
#  ENTRY POINT
# --------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()