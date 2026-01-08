import argparse
import os
import sys

import cv2
import numpy as np

def equalize_gray_custom(gray_u8: np.ndarray) -> np.ndarray:
    if gray_u8.dtype != np.uint8 or gray_u8.ndim != 2:
        raise ValueError("Ожидается grayscale uint8 изображение формы (H, W).")

    flat = gray_u8.ravel()
    hist = np.bincount(flat, minlength=256).astype(np.int64)
    cdf = np.cumsum(hist)
    n = flat.size

    nz = np.nonzero(hist)[0]
    if nz.size == 0:
        return gray_u8.copy()

    cdf_min = cdf[nz[0]]
    denom = n - cdf_min
    if denom <= 0:
        return gray_u8.copy()

    lut = ((cdf - cdf_min) * 255) // denom
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[gray_u8]

def equalize_custom(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = equalize_gray_custom(y)
        out = cv2.merge([y_eq, cr, cb])
        return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

    raise ValueError("Неподдерживаемый формат: нужен BGR.")


def equalize_opencv(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        out = cv2.merge([y_eq, cr, cb])
        return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

    raise ValueError("Неподдерживаемый формат: нужен BGR.")


def calc_metrics(custom: np.ndarray, opencv_res: np.ndarray) -> dict:
    mismatch = float(np.mean(custom != opencv_res) * 100.0)
    return {"mismatch_%": mismatch}


def make_side_by_side(*imgs_bgr_or_gray: np.ndarray) -> np.ndarray:
    prepared = []
    for im in imgs_bgr_or_gray:
        if im.ndim == 2:
            prepared.append(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
        else:
            prepared.append(im)

    h = min(im.shape[0] for im in prepared)
    resized = []
    for im in prepared:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            w = int(round(im.shape[1] * scale))
            resized.append(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA))
        else:
            resized.append(im)

    return cv2.hconcat(resized)


def put_text(im: np.ndarray, text: str, org=(10, 30)) -> np.ndarray:
    out = im.copy()
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Кастомная эквализация + сравнение с OpenCV.")
    ap.add_argument("image", help="Путь к изображению")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Не удалось открыть изображение: {args.image}", file=sys.stderr)
        sys.exit(1)

    if img.dtype != np.uint8:
        raise ValueError(f"Ожидается uint8 (8-bit). Сейчас: {img.dtype}")

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    custom = equalize_custom(img)
    opcv = equalize_opencv(img)

    metrics = calc_metrics(custom, opcv)
    print("Отклонение custom от OpenCV:")
    print(f"  mismatch_% : {metrics['mismatch_%']:.6f}%")

    # Склеим: исходная | custom | opencv |
    strip = make_side_by_side(img, custom, opcv)
    strip = put_text(
        strip,
        f"mismatch={metrics['mismatch_%']:.4f}%",
        org=(10, 30),
    )

    cv2.imshow("Original | Custom | OpenCV", strip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()