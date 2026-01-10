import argparse
import cv2
import numpy as np
import easyocr

from pathlib import Path


def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть файл: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - binary

    h, w = inv.shape
    kernel_len = max(40, w // 2)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 3))

    temp = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

    mask = cv2.dilate(temp, horiz_kernel, iterations=1)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    cleaned = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)

    cleaned = cv2.normalize(cleaned, None, 0, 255, cv2.NORM_MINMAX)

    scale = 3
    cleaned = cv2.resize(cleaned, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    src = Path(image_path)
    out_path = src.with_name(src.stem + "_preprocessed.png")
    cv2.imwrite(str(out_path), cleaned)
    print(f"Промежуточное изображение сохранено в: {out_path}")

    return cleaned


def image_to_text(image_path: str) -> str:
    img_clean = preprocess_image(image_path)
    result = reader.readtext(img_clean, detail=0, paragraph=True)
    
    return "\n".join(result).strip()


def main():
    parser = argparse.ArgumentParser(
        description="OCR текста с предварительной очисткой изображения."
    )
    parser.add_argument("image_path", help="Путь к файлу изображения")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.is_file():
        parser.error(f"Файл не найден: {img_path}")

    text = image_to_text(str(img_path))
    print(text)

reader = easyocr.Reader(['ch_sim'], gpu=False)

if __name__ == "__main__":
    main()