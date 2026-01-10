import cv2 as cv
import numpy as np
import argparse
import os


def get_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Диапазон для розовых тонов подбирался под картинку
    lower_pink = np.array([140, 120, 120], dtype=np.uint8)
    upper_pink = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv.inRange(hsv, lower_pink, upper_pink)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel, iterations=1)

    return mask


def main():
    parser = argparse.ArgumentParser(description="Сегментация штанов по цвету")
    parser.add_argument("image", help="Путь к исходному изображению")
    parser.add_argument("output", help="Путь для сохранения маски (по умолчанию <имя>_mask.png)")
    args = parser.parse_args()

    img = cv.imread(args.image)
    if img is None:
        raise SystemExit(f"Не удалось открыть изображение: {args.image}")

    mask = get_mask(img)

    if args.output:
        out_path = args.output
    else:
        root, _ = os.path.splitext(args.image)
        out_path = root + "_mask.png"

    cv.imwrite(out_path, mask)
    print(f"Маска сохранена в: {out_path}")

if __name__ == "__main__":
    main()