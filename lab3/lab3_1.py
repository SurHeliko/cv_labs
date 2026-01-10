import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn


def rotate_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
    (h, w) = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def nonblack_mask(img_bgr: np.ndarray, thr: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def estimate_rotation_angle(img_bgr: np.ndarray) -> float:
    mask = nonblack_mask(img_bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    angle = rect[2]
    if angle < -45:
        angle += 90
    return angle


def crop_to_content(img_bgr: np.ndarray, pad: int = 3) -> np.ndarray:
    mask = nonblack_mask(img_bgr)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_bgr
    x1 = max(int(xs.min()) - pad, 0)
    y1 = max(int(ys.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad, img_bgr.shape[1] - 1)
    y2 = min(int(ys.max()) + pad, img_bgr.shape[0] - 1)
    return img_bgr[y1:y2 + 1, x1:x2 + 1].copy()


class BuiltFromArch(nn.Module):
    def __init__(self, arch: list[dict]):
        super().__init__()
        modules = []
        for layer in arch:
            t = layer["type"]
            if t == "pad":
                modules.append(nn.ZeroPad2d(tuple(layer["pad"])))
            elif t == "conv":
                modules.append(nn.Conv2d(
                    in_channels=layer["in_ch"],
                    out_channels=layer["out_ch"],
                    kernel_size=tuple(layer["ks"]),
                    stride=tuple(layer["stride"]),
                    padding=tuple(layer["padding"]),
                    dilation=tuple(layer["dilation"]),
                    bias=layer["bias"],
                ))
            elif t == "relu":
                modules.append(nn.ReLU(inplace=True))
            elif t == "bnorm":
                modules.append(nn.BatchNorm2d(layer["num_features"], eps=layer["eps"], momentum=0.1, affine=True))
            else:
                raise RuntimeError(f"Неизвестный слой в arch: {t}")
        self.seq = nn.Sequential(*modules)


    def forward(self, x):
        return self.seq(x)


def load_model_pth(pth_path: str, device: torch.device) -> tuple[nn.Module, str]:
    obj = torch.load(pth_path, map_location="cpu")

    if isinstance(obj, nn.Module):
        return obj.to(device).eval(), "residual"

    if isinstance(obj, dict) and "arch" in obj and "state_dict" in obj:
        model = BuiltFromArch(obj["arch"])
        model.load_state_dict(obj["state_dict"], strict=True)
        model.to(device).eval()
        return model, obj.get("default_model_output", "residual")

    raise RuntimeError("Не понял формат .pth (ожидаю nn.Module или dict с arch/state_dict).")


def first_conv_in_channels(model: nn.Module) -> int:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return int(m.in_channels)
    raise RuntimeError("В модели не найден Conv2d")


def pixel_unshuffle(x: torch.Tensor, r: int = 2) -> torch.Tensor:
    n, c, h, w = x.shape
    if h % r != 0 or w % r != 0:
        raise RuntimeError(f"pixel_unshuffle требует H,W кратные {r}, а сейчас {h}x{w}")
    x = x.view(n, c, h // r, r, w // r, r)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(n, c * r * r, h // r, w // r)


def pixel_shuffle(x: torch.Tensor, r: int = 2) -> torch.Tensor:
    n, c, h, w = x.shape
    if c % (r * r) != 0:
        raise RuntimeError(f"pixel_shuffle требует C кратный {r*r}, а сейчас C={c}")
    oc = c // (r * r)
    x = x.view(n, oc, r, r, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(n, oc, h * r, w * r)


def make_weight(h: int, w: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((h, w), dtype=np.float32)
    ov_y = min(overlap, h // 2)
    ov_x = min(overlap, w // 2)
    wx = np.ones(w, dtype=np.float32)
    wy = np.ones(h, dtype=np.float32)
    if ov_x > 0:
        ramp = np.linspace(0.0, 1.0, ov_x, dtype=np.float32)
        wx[:ov_x] = ramp
        wx[-ov_x:] = ramp[::-1]
    if ov_y > 0:
        ramp = np.linspace(0.0, 1.0, ov_y, dtype=np.float32)
        wy[:ov_y] = ramp
        wy[-ov_y:] = ramp[::-1]
    return wy[:, None] * wx[None, :]


def tile_starts(L: int, tile: int, stride: int) -> list[int]:
    if L <= tile:
        return [0]
    s = list(range(0, L - tile + 1, stride))
    if s[-1] != L - tile:
        s.append(L - tile)
    return s


@torch.no_grad()
def denoise_rgb_u8(
    rgb_u8: np.ndarray,
    model: nn.Module,
    device: torch.device,
    sigma: float,
    model_output: str,
    model_input_range: str,
    tile: int,
    overlap: int,
) -> np.ndarray:
    assert rgb_u8.dtype == np.uint8 and rgb_u8.ndim == 3 and rgb_u8.shape[2] == 3

    in_ch = first_conv_in_channels(model)
    if in_ch == 3:
        mode = "rgb3"
    elif in_ch == 4:
        mode = "rgb_sigma4"
    elif in_ch == 16:
        mode = "ffdnet16"
        tile = max(64, (tile // 2) * 2)
        overlap = max(0, (overlap // 2) * 2)
    else:
        raise RuntimeError(f"Неожиданный in_ch первой conv: {in_ch} (поддерживается 3/4/16)")

    if model_input_range == "0_1":
        img = rgb_u8.astype(np.float32) / 255.0
        sigma_val = float(sigma) / 255.0
        def to_u8(t): return (t.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    else:
        img = rgb_u8.astype(np.float32)
        sigma_val = float(sigma)
        def to_u8(t): return t.clamp(0.0, 255.0).round().to(torch.uint8)

    H, W = img.shape[:2]
    stride = max(1, tile - 2 * overlap)
    ys = tile_starts(H, tile, stride)
    xs = tile_starts(W, tile, stride)

    out = torch.zeros((1, 3, H, W), dtype=torch.float32, device=device)
    wsum = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)

    for y0 in ys:
        y1 = min(H, y0 + tile)
        for x0 in xs:
            x1 = min(W, x0 + tile)

            patch = img[y0:y1, x0:x1, :]
            th, tw = patch.shape[:2]

            w = make_weight(th, tw, overlap)
            w_t = torch.from_numpy(w).to(device=device, dtype=torch.float32)[None, None, :, :]

            x = torch.from_numpy(patch).to(device=device, dtype=torch.float32)
            x = x.permute(2, 0, 1).unsqueeze(0)

            if mode == "rgb3":
                pred = model(x)
                clean = (x - pred) if (model_output == "residual") else pred

            elif mode == "rgb_sigma4":
                sigma_map = torch.full((1, 1, th, tw), sigma_val, device=device, dtype=torch.float32)
                xin = torch.cat([x, sigma_map], dim=1)
                pred = model(xin)
                clean = (x - pred) if (model_output == "residual") else pred

            else:
                pad_b = th % 2
                pad_r = tw % 2
                x2 = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b), mode="reflect") if (pad_b or pad_r) else x

                th2, tw2 = x2.shape[-2], x2.shape[-1]
                sigma_map = torch.full((1, 1, th2, tw2), sigma_val, device=device, dtype=torch.float32)

                x_un = pixel_unshuffle(x2, 2)
                s_un = pixel_unshuffle(sigma_map, 2)
                xin = torch.cat([x_un, s_un], dim=1)

                pred = model(xin)
                if pred.shape[1] == 12:
                    clean_un = (x_un - pred) if (model_output == "residual") else pred
                    clean2 = pixel_shuffle(clean_un, 2)
                elif pred.shape[1] == 3:
                    clean2 = (x2 - pred) if (model_output == "residual") else pred
                else:
                    raise RuntimeError(f"FFDNet: неожиданный выход C={pred.shape[1]} (ожидал 12 или 3)")

                clean = clean2[:, :, :th, :tw]

            out[:, :, y0:y1, x0:x1] += clean * w_t
            wsum[:, :, y0:y1, x0:x1] += w_t

    out = out / torch.clamp(wsum, min=1e-6)
    out_u8 = to_u8(out).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return out_u8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Входное изображение")
    ap.add_argument("output", help="Выходное изображение")
    ap.add_argument("model", help="Путь к .pth")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--sigma", type=float, default=50.0, help="Уровень шума (0..255). Для сильных точек пробуйте 50/75")
    ap.add_argument("--model-output", choices=["residual", "direct"], default="direct")
    ap.add_argument("--model-input-range", choices=["0_1", "0_255"], default="0_1")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=32)
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not os.path.isfile(args.model):
        raise SystemExit(f"Не найден файл модели: {args.model}")

    model, default_out = load_model_pth(args.model, device=device)

    img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Не удалось прочитать: {args.input}")

    angle = estimate_rotation_angle(img_bgr)
    aligned = rotate_bound(img_bgr, angle_deg=angle)
    aligned = crop_to_content(aligned, pad=3)

    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    rgb_dn = denoise_rgb_u8(
        rgb, model, device=device,
        sigma=args.sigma,
        model_output=args.model_output,
        model_input_range=args.model_input_range,
        tile=args.tile,
        overlap=args.overlap,
    )

    out_bgr = cv2.cvtColor(rgb_dn, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(args.output, out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise SystemExit(f"Не удалось записать: {args.output}")


if __name__ == "__main__":
    main()