"""
Filename: rtod_v1.py
Version: 1 — Image-based Object Measurement (dengan File Dialog)
Usage:
    Jalankan → popup pilih foto → hasil tampil otomatis.
    Referensi: Lingkaran diameter 3cm + Persegi 3x3cm (di sisi kiri gambar).
"""

import sys
import numpy as np
import cv2
import imutils
from imutils import perspective, contours
from scipy.spatial.distance import euclidean

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

# ─────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────
REF_CIRCLE_DIAM = 3.0   # cm
REF_SQUARE_SIDE = 3.0   # cm


# ─────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def is_circle(cnt, thresh=0.80):
    area  = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim == 0:
        return False
    return (4 * np.pi * area) / (perim ** 2) >= thresh


def is_square(cnt, tol=0.15):
    peri  = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) != 4:
        return False
    x, y, w, h = cv2.boundingRect(approx)
    return abs(1.0 - w / float(h)) <= tol


def compute_scale(cnts_sorted):
    circle_idx = square_idx = None
    for i, c in enumerate(cnts_sorted):
        if circle_idx is None and is_circle(c):
            circle_idx = i
        elif square_idx is None and is_square(c):
            square_idx = i
        if circle_idx is not None and square_idx is not None:
            break

    ratios = []
    if circle_idx is not None:
        _, radius = cv2.minEnclosingCircle(cnts_sorted[circle_idx])
        ratios.append((2 * radius) / REF_CIRCLE_DIAM)
    if square_idx is not None:
        box = cv2.minAreaRect(cnts_sorted[square_idx])
        pts = perspective.order_points(np.array(cv2.boxPoints(box), dtype="int"))
        tl, tr, br, _ = pts
        side = (euclidean(tl, tr) + euclidean(tr, br)) / 2
        ratios.append(side / REF_SQUARE_SIDE)

    ref_idx = [i for i in (circle_idx, square_idx) if i is not None]
    return (float(np.mean(ratios)), ref_idx) if ratios else (None, [])


def draw_measurement(image, cnt, ppc, is_ref=False):
    box = cv2.minAreaRect(cnt)
    pts = perspective.order_points(np.array(cv2.boxPoints(box), dtype="int"))
    tl, tr, br, bl = pts
    color = (0, 165, 255) if is_ref else (50, 205, 50)
    cv2.drawContours(image, [pts.astype("int")], -1, color, 2)

    mid_t = midpoint(tl, tr)
    mid_r = midpoint(tr, br)
    wid = euclidean(tl, tr) / ppc
    ht  = euclidean(tr, br) / ppc

    cv2.putText(image, f"{wid:.1f}cm",
                (int(mid_t[0] - 20), int(mid_t[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 0), 2)
    cv2.putText(image, f"{ht:.1f}cm",
                (int(mid_r[0] + 8), int(mid_r[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 0), 2)


def process_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None, "Gagal membaca gambar."

    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (9, 9), 0)
    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    raw_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_cnts = imutils.grab_contours(raw_cnts)
    (cnts_sorted, _) = contours.sort_contours(raw_cnts)
    cnts_sorted = [c for c in cnts_sorted if cv2.contourArea(c) > 100]

    if not cnts_sorted:
        return None, "Tidak ada kontur yang terdeteksi."

    ppc, ref_idx = compute_scale(cnts_sorted)
    if ppc is None:
        cv2.putText(image,
                    "REFERENSI tidak terdeteksi! Pastikan lingkaran+persegi 3cm di kiri.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return image, "Referensi tidak ditemukan."

    for i, cnt in enumerate(cnts_sorted):
        draw_measurement(image, cnt, ppc, is_ref=(i in ref_idx))

    # HUD overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (430, 65), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    cv2.putText(image, f"Skala: {ppc:.2f} px/cm",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
    cv2.putText(image, "ORANGE = Referensi   |   HIJAU = Objek",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

    return image, f"OK — {ppc:.2f} px/cm"


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    img_path, _ = QFileDialog.getOpenFileName(
        None,
        "Pilih Gambar untuk Diukur",
        "",
        "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff);;All Files (*)"
    )

    if not img_path:
        QMessageBox.information(None, "Info", "Tidak ada gambar yang dipilih. Program keluar.")
        sys.exit(0)

    result_img, status = process_image(img_path)

    if result_img is None:
        QMessageBox.critical(None, "Error", status)
        sys.exit(1)

    # Resize jika terlalu besar
    h, w = result_img.shape[:2]
    if max(h, w) > 950:
        scale = 950 / max(h, w)
        result_img = cv2.resize(result_img, (int(w * scale), int(h * scale)))

    cv2.imshow("Object Measurement v1  —  Tekan tombol apa saja untuk tutup", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(0)


if __name__ == "__main__":
    main()