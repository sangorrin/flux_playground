# juillard_panels_v3.py
# Split comic pages into panel crops using gutters (white margins) only.
# Output: a flat dataset folder with sequential files 0001.jpg, 0002.jpg, ...
# Optional: save debug pages with red rectangles to a separate folder.
#
# Usage examples:
#   python juillard_panels_v3.py \
#       "./pages" -o dataset --square --debug
#
#   python juillard_panels_v3.py \
#       "page001.jpg" -o dataset --start 101 --jpg-quality 92 --debug

import cv2
import numpy as np
from pathlib import Path
import argparse

# ---------- util ----------
def to_hsv_white_mask(bgr, s_max=40, v_min=180):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    white = ((S <= s_max) & (V >= v_min)).astype(np.uint8) * 255
    return white


def flood_border_white(mask):  # return mask with only white connected to page border at 255
    h, w = mask.shape
    m = mask.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    # top/bottom
    for x in range(w):
        if m[0, x] == 255:
            cv2.floodFill(m, ffmask, (x, 0), 200)
        if m[h - 1, x] == 255:
            cv2.floodFill(m, ffmask, (x, h - 1), 200)
    # left/right
    for y in range(h):
        if m[y, 0] == 255:
            cv2.floodFill(m, ffmask, (0, y), 200)
        if m[y, w - 1] == 255:
            cv2.floodFill(m, ffmask, (w - 1, y), 200)
    m = (m == 200).astype(np.uint8) * 255
    return m


def merge_overlaps(boxes, iou_thresh=0.20):
    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / float(union)

    out = []
    for b in sorted(boxes, key=lambda r: r[2] * r[3], reverse=True):
        fused = False
        for i, m in enumerate(out):
            if iou(b, m) > iou_thresh:
                x = min(b[0], m[0])
                y = min(b[1], m[1])
                w = max(b[0] + b[2], m[0] + m[2]) - x
                h = max(b[1] + b[3], m[1] + m[3]) - y
                out[i] = (x, y, w, h)
                fused = True
                break
        if not fused:
            out.append(b)
    return out


def order_reading(boxes, bucket=60):  # row → column ordering
    return sorted(boxes, key=lambda r: (round(r[1] / bucket) * bucket, r[0]))


# ---------- A) Panels by gutters (white margins) ----------
def panels_by_gutters(img, s_max=40, v_min=180, dilate=6, min_area=25000, pad=8):
    H, W = img.shape[:2]
    white = to_hsv_white_mask(img, s_max, v_min)
    gutters = flood_border_white(white)  # white connected to outer border
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
    gutters = cv2.dilate(gutters, k, iterations=1)  # thicken gutters to guarantee splits

    # content = everything that's not a gutter
    content = 255 - gutters
    num, labels, stats, _ = cv2.connectedComponentsWithStats(content, connectivity=8)

    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if w * h < min_area:
            continue
        if w * h > 0.97 * W * H:  # skip whole page
            continue
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        rect_score = comp.sum() / float(w * h)  # rectangularity proxy
        if rect_score < 0.55:
            continue
        xx = max(0, x - pad)
        yy = max(0, y - pad)
        ww = min(W, x + w + pad) - xx
        hh = min(H, y + h + pad) - yy
        boxes.append((xx, yy, ww, hh))

    return order_reading(merge_overlaps(boxes, 0.2))


# ---------- B) Fallback by page grid lines (Hough) ----------
def panels_by_lines(img, white_gutters, min_area=25000, pad=6, merge_tol=12):
    H, W = img.shape[:2]
    edges = cv2.Canny(white_gutters, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120,
        minLineLength=min(H, W) // 3, maxLineGap=10
    )
    if lines is None:
        return []

    vpos, hpos = [], []
    for (x1, y1, x2, y2) in lines[:, 0]:
        if abs(x1 - x2) < 5:
            vpos.append(x1)
        if abs(y1 - y2) < 5:
            hpos.append(y1)

    def cluster(pos):
        pos = sorted(pos)
        out = []
        for p in pos:
            if not out or abs(p - out[-1]) > merge_tol:
                out.append(p)
        return out

    xs = cluster([0, W - 1] + vpos)
    ys = cluster([0, H - 1] + hpos)

    boxes = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x, y = xs[i], ys[j]
            w, h = xs[i + 1] - x, ys[j + 1] - y
            if w * h < min_area:
                continue
            per = 0
            per += (white_gutters[y, x:x + w] == 255).mean()
            per += (white_gutters[y + h - 1, x:x + w] == 255).mean()
            per += (white_gutters[y:y + h, x] == 255).mean()
            per += (white_gutters[y:y + h, x + w - 1] == 255).mean()
            if per / 4.0 < 0.5:
                continue
            xx = max(0, x + pad)
            yy = max(0, y + pad)
            ww = max(1, w - 2 * pad)
            hh = max(1, h - 2 * pad)
            boxes.append((xx, yy, ww, hh))
    return order_reading(merge_overlaps(boxes, 0.15))


# ---------- saving ----------
def save_crop_jpg(img, box, out_dir, seq_idx, square=False,
                  min_side=512, max_side=1024, jpg_quality=95):
    x, y, w, h = box
    crop = img[y:y + h, x:x + w]

    if square:
        side = max(w, h)
        canvas = np.full((side, side, 3), 255, np.uint8)
        ox = (side - w) // 2
        oy = (side - h) // 2
        canvas[oy:oy + h, ox:ox + w] = crop
        crop = canvas

    H, W = crop.shape[:2]
    s = 1.0
    if min(H, W) < min_side:
        s = min_side / min(H, W)
    if max(H * s, W * s) > max_side:
        s = max_side / max(H, W)
    if s != 1.0:
        crop = cv2.resize(crop, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seq_idx:04d}.jpg"
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)])


def process_page(path, args, seq_start, page_idx):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read {path}")

    H, W = img.shape[:2]
    boxes = panels_by_gutters(
        img,
        s_max=args.white_s_max,
        v_min=args.white_v_min,
        dilate=args.gutter_dilate,
        min_area=args.min_area,
        pad=args.pad,
    )

    if (len(boxes) <= 2) or (sum(w * h for x, y, w, h in boxes) < 0.5 * W * H):
        white = to_hsv_white_mask(img, args.white_s_max, args.white_v_min)
        gutters = flood_border_white(white)
        boxes2 = panels_by_lines(img, gutters, args.min_area, args.pad, args.merge_tol)
        if len(boxes2) > len(boxes):
            boxes = boxes2

    # Debug image
    if args.debug:
        dbg = img.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
        (Path(args.output) / args.debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(args.output) / args.debug_dir / f"debug_boxes_page_{page_idx}.jpg"), dbg)

    # Save crops with global sequential numbering
    seq = seq_start
    for b in boxes:
        save_crop_jpg(
            img,
            b,
            Path(args.output),
            seq,
            square=args.square,
            min_side=args.min_side,
            max_side=args.max_side,
            jpg_quality=args.jpg_quality,
        )
        seq += 1
    return seq, len(boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Image or folder of pages")
    ap.add_argument("-o", "--output", default="dataset", help="Output dataset folder")
    ap.add_argument("--start", type=int, default=1, help="Sequential start index (default 1 → 0001.jpg)")
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--min-area", type=int, default=25000)
    ap.add_argument("--square", action="store_true")
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--jpg-quality", type=int, default=95)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-dir", default="debug", help="Subfolder inside output for debug pages")
    # gutters/lines params
    ap.add_argument("--white-s-max", type=int, default=40, help="HSV S max for white")
    ap.add_argument("--white-v-min", type=int, default=180, help="HSV V min for white")
    ap.add_argument("--gutter-dilate", type=int, default=6, help="Thicken gutters (pixels)")
    ap.add_argument("--merge-tol", type=int, default=12, help="Hough lines merge tolerance")

    args = ap.parse_args()

    p = Path(args.input)
    if p.is_dir():
        imgs = sorted([*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png"), *p.glob("*.tif"), *p.glob("*.webp")])
    else:
        imgs = [p]

    seq = args.start
    total_panels = 0
    for page_idx, f in enumerate(imgs, start=1):
        try:
            seq, n = process_page(f, args, seq, page_idx)
            total_panels += n
        except Exception as e:
            print(f"[WARN] {f}: {e}")

    print(f"[OK] {len(imgs)} page(s) | {total_panels} panels → {args.output} starting at {args.start:04d}")


if __name__ == "__main__":
    main()
