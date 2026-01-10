import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn


def load_mat_any(path: str) -> dict:
    try:
        import scipy.io as sio
        return sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        import mat73
        return mat73.loadmat(path)


def get_field(obj, name: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    if hasattr(obj, name):
        return getattr(obj, name)
    try:
        if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
            return obj[name]
    except Exception:
        pass
    return None


def to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return list(x.flat)
    return [x]


def _extract_conv_weights_bias(weights):
    if isinstance(weights, np.ndarray) and weights.dtype == object:
        weights = list(weights.flat)
    if not isinstance(weights, (list, tuple)):
        raise RuntimeError("conv.weights: неожиданный формат (ожидался list/tuple)")
    w = weights[0]
    b = weights[1] if len(weights) > 1 else None
    return w, b


def _matconvnet_filter_to_torch(w: np.ndarray) -> np.ndarray:
    if w.ndim != 4:
        raise RuntimeError(f"Ожидался 4D фильтр, получено {w.shape}")
    return np.transpose(w, (3, 2, 0, 1)).copy()


def _pair(v, default):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return (int(v), int(v))
    v = np.array(v).reshape(-1).tolist()
    if len(v) == 1:
        return (int(v[0]), int(v[0]))
    return (int(v[0]), int(v[1]))


def _prefix_state_dict(sd: dict, prefix: str) -> dict:
    if not sd:
        return sd
    if all(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {f"{prefix}{k}": v for k, v in sd.items()}


def build_arch_and_state_from_mat(net_obj):
    layers = get_field(net_obj, "layers")
    if layers is None:
        raise RuntimeError(
            "В .mat не найдено net.layers. Этот конвертер поддерживает MatConvNet SimpleNN.\n"
            "Если у вас DagNN-граф, структура другая."
        )

    layers_list = to_list(layers)

    type_counts: dict[str, int] = {}
    for layer in layers_list:
        ltype = get_field(layer, "type")
        if isinstance(ltype, bytes):
            ltype = ltype.decode("utf-8", errors="ignore")
        ltype = str(ltype)
        type_counts[ltype] = type_counts.get(ltype, 0) + 1

    arch = []
    skipped = []
    state_model = nn.Sequential()

    cur_ch = None
    seen_first_conv = False

    for i, layer in enumerate(layers_list):
        ltype = get_field(layer, "type")
        if isinstance(ltype, bytes):
            ltype = ltype.decode("utf-8", errors="ignore")
        ltype = str(ltype)

        if ltype == "concat":
            skipped.append({"index": i, "type": "concat"})
            continue

        if ltype == "conv":
            weights = get_field(layer, "weights")
            w, b = _extract_conv_weights_bias(weights)

            w = np.array(w, dtype=np.float32)
            w_t = _matconvnet_filter_to_torch(w)
            out_ch, in_ch, kh, kw = w_t.shape

            stride = _pair(get_field(layer, "stride"), (1, 1))
            dilate = _pair(get_field(layer, "dilate"), (1, 1))
            pad = get_field(layer, "pad")

            pad_layer = None
            conv_padding = (kh // 2, kw // 2)

            if pad is not None:
                p = np.array(pad).reshape(-1).tolist()
                if len(p) == 4:
                    top, bottom, left, right = map(int, p)
                    if (top != bottom) or (left != right):
                        pad_layer = (left, right, top, bottom)  # (l,r,t,b)
                        conv_padding = (0, 0)
                    else:
                        conv_padding = (top, left)
                elif len(p) == 2:
                    conv_padding = (int(p[0]), int(p[1]))
                elif len(p) == 1:
                    conv_padding = (int(p[0]), int(p[0]))

            if seen_first_conv and cur_ch is not None and in_ch != cur_ch:
                raise RuntimeError(
                    f"Сеть не выглядит последовательной: conv слой {i} ожидает in_ch={in_ch}, "
                    f"а предыдущий выход был {cur_ch}.\n"
                    "Этот конвертер поддерживает только последовательные сети, где concat применяется к входу."
                )

            if pad_layer is not None:
                arch.append({"type": "pad", "pad": list(pad_layer)})
                state_model.append(nn.ZeroPad2d(pad_layer))

            conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=(kh, kw),
                stride=stride,
                padding=conv_padding,
                dilation=dilate,
                bias=True,
            )
            conv.weight.data = torch.from_numpy(w_t)
            if b is None:
                conv.bias.data.zero_()
            else:
                b = np.array(b, dtype=np.float32).reshape(-1)
                conv.bias.data = torch.from_numpy(b)

            arch.append({
                "type": "conv",
                "in_ch": in_ch,
                "out_ch": out_ch,
                "ks": [kh, kw],
                "stride": list(stride),
                "padding": list(conv_padding),
                "dilation": list(dilate),
                "bias": True,
            })
            state_model.append(conv)

            seen_first_conv = True
            cur_ch = out_ch
            continue

        if ltype == "relu":
            arch.append({"type": "relu"})
            state_model.append(nn.ReLU(inplace=True))
            continue

        if ltype == "bnorm":
            weights = get_field(layer, "weights")
            if isinstance(weights, np.ndarray) and weights.dtype == object:
                weights = list(weights.flat)
            if not isinstance(weights, (list, tuple)) or len(weights) < 3:
                raise RuntimeError("bnorm слой: ожидаю weights=[gamma,beta,moments]")

            gamma = np.array(weights[0], dtype=np.float32).reshape(-1)
            beta = np.array(weights[1], dtype=np.float32).reshape(-1)
            moments = np.array(weights[2], dtype=np.float32)

            if moments.ndim == 2 and moments.shape[0] == 2:
                mean = moments[0].reshape(-1)
                var = moments[1].reshape(-1)
            elif moments.ndim == 2 and moments.shape[1] == 2:
                mean = moments[:, 0].reshape(-1)
                var = moments[:, 1].reshape(-1)
            else:
                raise RuntimeError(f"Не понял moments формы {moments.shape}")

            eps = float(get_field(layer, "epsilon") or 1e-5)
            C = int(gamma.shape[0])

            if cur_ch is not None and C != cur_ch:
                raise RuntimeError(
                    f"bnorm на слое {i}: C={C}, но текущие каналы {cur_ch}. "
                    "Похоже на не-последовательную сеть."
                )

            bn = nn.BatchNorm2d(C, eps=eps, momentum=0.1, affine=True, track_running_stats=True)
            bn.weight.data = torch.from_numpy(gamma)
            bn.bias.data = torch.from_numpy(beta)
            bn.running_mean.data = torch.from_numpy(mean)
            bn.running_var.data = torch.from_numpy(var)

            arch.append({"type": "bnorm", "num_features": C, "eps": eps})
            state_model.append(bn)
            continue

        raise RuntimeError(f"Неподдерживаемый слой в .mat: type={ltype}")

    sd_raw = state_model.state_dict()
    sd = _prefix_state_dict(sd_raw, "seq.")

    return arch, sd, type_counts, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Путь к FDnCNN_color.mat")
    ap.add_argument("outtput", help="Куда сохранить .pth")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Не найден файл: {args.input}")

    md = load_mat_any(args.input)
    net = md.get("net") or md.get("Net") or md.get("model") or md.get("Model")
    if net is None:
        raise SystemExit(f"В .mat не найден объект 'net'. Ключи: {list(md.keys())[:40]}")

    arch, sd, type_counts, skipped = build_arch_and_state_from_mat(net)

    payload = {
        "format": "matconvnet_simplenn_to_torch",
        "source_mat": os.path.basename(args.input),
        "arch": arch,
        "state_dict": sd,
        "default_model_output": "residual",
        "layer_type_counts": type_counts,
        "skipped_layers": skipped,
        "note": "concat слои пропущены; state_dict сохранён с префиксом 'seq.' чтобы совпадать с BuiltFromArch(self.seq).",
    }

    out_abs = os.path.abspath(args.outtput)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    torch.save(payload, out_abs)

if __name__ == "__main__":
    main()