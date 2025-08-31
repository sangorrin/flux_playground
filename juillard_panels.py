# juillard_panels_v3.py
import cv2, numpy as np
from pathlib import Path
import argparse

# ---------- util ----------
def to_hsv_white_mask(bgr, s_max=40, v_min=180):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:,:,1], hsv[:,:,2]
    white = ((S <= s_max) & (V >= v_min)).astype(np.uint8)*255
    return white

def flood_border_white(mask):  # deja a 255 solo el blanco conectado al borde
    h,w = mask.shape
    m = mask.copy()
    ffmask = np.zeros((h+2,w+2), np.uint8)
    # arriba/abajo
    for x in range(w):
        if m[0,x]==255:      cv2.floodFill(m, ffmask, (x,0), 200)
        if m[h-1,x]==255:    cv2.floodFill(m, ffmask, (x,h-1), 200)
    # izq/der
    for y in range(h):
        if m[y,0]==255:      cv2.floodFill(m, ffmask, (0,y), 200)
        if m[y,w-1]==255:    cv2.floodFill(m, ffmask, (w-1,y), 200)
    m = (m==200).astype(np.uint8)*255
    return m

def merge_overlaps(boxes, iou_thresh=0.20):
    def iou(a,b):
        ax,ay,aw,ah=a; bx,by,bw,bh=b
        x1=max(ax,bx); y1=max(ay,by)
        x2=min(ax+aw,bx+bw); y2=min(ay+ah,by+bh)
        if x2<=x1 or y2<=y1: return 0
        inter=(x2-x1)*(y2-y1)
        return inter/float(aw*ah + bw*bh - inter)
    out=[]
    for b in sorted(boxes, key=lambda r:r[2]*r[3], reverse=True):
        fused=False
        for i,m in enumerate(out):
            if iou(b,m)>iou_thresh:
                x=min(b[0],m[0]); y=min(b[1],m[1])
                w=max(b[0]+b[2],m[0]+m[2])-x
                h=max(b[1]+b[3],m[1]+m[3])-y
                out[i]=(x,y,w,h); fused=True; break
        if not fused: out.append(b)
    return out

def order_reading(boxes,b=60):  # fila→columna
    return sorted(boxes, key=lambda r:(round(r[1]/b)*b, r[0]))

# ---------- A) Split por gutters blancos ----------
def panels_by_gutters(img, s_max=40, v_min=180, dilate=6, min_area=25000, pad=8):
    H,W = img.shape[:2]
    white = to_hsv_white_mask(img, s_max, v_min)
    # blanco conectado al borde = gutters
    gutters = flood_border_white(white)
    # engrosar gutter para cortar seguro
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate,dilate))
    gutters = cv2.dilate(gutters, k, iterations=1)
    # contenido = lo que NO es gutter
    content = 255 - gutters
    num, labels, stats, _ = cv2.connectedComponentsWithStats(content, connectivity=8)
    boxes=[]
    for i in range(1,num):
        x,y,w,h,area = stats[i]
        if w*h < min_area: continue
        if w*h > 0.97*W*H: continue
        # rectangularidad (área componente vs caja)
        comp = (labels[y:y+h, x:x+w]==i).astype(np.uint8)
        rect_score = comp.sum() / float(w*h)
        if rect_score < 0.55: continue
        # padding
        xx=max(0,x-pad); yy=max(0,y-pad)
        ww=min(W,x+w+pad)-xx; hh=min(H,y+h+pad)-yy
        boxes.append((xx,yy,ww,hh))
    return order_reading(merge_overlaps(boxes,0.2))

# ---------- B) Fallback por líneas (Hough) ----------
def panels_by_lines(img, white_gutters, min_area=25000, pad=6, merge_tol=12):
    H,W = img.shape[:2]
    edges = cv2.Canny(white_gutters, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=min(H,W)//3, maxLineGap=10)
    if lines is None: return []
    vpos, hpos = [], []
    for (x1,y1,x2,y2) in lines[:,0]:
        if abs(x1-x2) < 5: vpos.append(x1)
        if abs(y1-y2) < 5: hpos.append(y1)
    def cluster(pos):
        pos = sorted(pos)
        out=[]
        for p in pos:
            if not out or abs(p-out[-1])>merge_tol: out.append(p)
        return out
    xs = cluster([0,W-1]+vpos)
    ys = cluster([0,H-1]+hpos)

    boxes=[]
    # generamos rects entre líneas consecutivas y validamos
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            x,y = xs[i], ys[j]
            w,h = xs[i+1]-x, ys[j+1]-y
            if w*h < min_area: continue
            # check: borde debe ser mostly gutter
            per = 0
            per += (white_gutters[y, x:x+w]==255).mean()
            per += (white_gutters[y+h-1, x:x+w]==255).mean()
            per += (white_gutters[y:y+h, x]==255).mean()
            per += (white_gutters[y:y+h, x+w-1]==255).mean()
            if per/4.0 < 0.5:  # la mitad del borde blanco
                continue
            xx=max(0,x+pad); yy=max(0,y+pad)
            ww=max(1,w-2*pad); hh=max(1,h-2*pad)
            boxes.append((xx,yy,ww,hh))
    return order_reading(merge_overlaps(boxes,0.15))

# ---------- Inpaint bocadillos ----------
def inpaint_balloons(panel, min_rel=0.01, max_rel=0.32, strength=3):
    h,w = panel.shape[:2]; A=h*w
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    white = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)[1]
    # quitar blanco conectado a borde del panel
    wb = flood_border_white(white)
    inner = cv2.subtract(white, wb)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    inner = cv2.morphologyEx(inner, cv2.MORPH_CLOSE, ker, iterations=1)
    cnts,_ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h,w), np.uint8)
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_rel*A <= area <= max_rel*A): continue
        peri = cv2.arcLength(c, True)
        if peri<10: continue
        circ = 4*np.pi*area/(peri*peri)
        hull = cv2.convexHull(c)
        solidity = area/(cv2.contourArea(hull)+1e-5)
        if circ>=0.25 and solidity>=0.85:
            cv2.drawContours(mask,[c],-1,255,cv2.FILLED)
    if mask.sum()==0: return panel
    mask = cv2.dilate(mask, ker, 1)
    return cv2.inpaint(panel, mask, strength, cv2.INPAINT_TELEA)

# ---------- Pipeline ----------
def save_crop(img, box, outdir, i, square=False, min_side=512, max_side=1024):
    x,y,w,h = box
    crop = img[y:y+h, x:x+w]
    if square:
        side = max(w,h)
        canvas = np.full((side,side,3), 255, np.uint8)
        ox=(side-w)//2; oy=(side-h)//2
        canvas[oy:oy+h, ox:ox+w] = crop
        crop = canvas
    H,W = crop.shape[:2]
    s=1.0
    if min(H,W)<min_side: s=min_side/min(H,W)
    if max(H*s,W*s)>max_side: s=max_side/max(H,W)
    if s!=1.0:
        crop = cv2.resize(crop, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir/f"panel_{i:03d}.png"), crop)
    return crop

def process_one(path, args):
    img = cv2.imread(str(path))
    H,W = img.shape[:2]
    boxes = panels_by_gutters(img, s_max=args.white_s_max, v_min=args.white_v_min,
                              dilate=args.gutter_dilate, min_area=args.min_area, pad=args.pad)
    # fallback si salen muy pocas / muy grandes
    if (len(boxes)<=2) or (sum(w*h for x,y,w,h in boxes) < 0.5*W*H):
        white = to_hsv_white_mask(img, args.white_s_max, args.white_v_min)
        gutters = flood_border_white(white)
        boxes2 = panels_by_lines(img, gutters, args.min_area, args.pad, args.merge_tol)
        if len(boxes2)>len(boxes): boxes = boxes2

    base = Path(args.output)/Path(path).stem
    dbg = img.copy()
    for i,b in enumerate(boxes,1):
        crop = save_crop(img,b,base,i,args.square,args.min_side,args.max_side)
        if args.inpaint:
            clean = inpaint_balloons(crop, args.balloon_min_rel, args.balloon_max_rel, args.inpaint_strength)
            cv2.imwrite(str(base/f"panel_{i:03d}_clean.png"), clean)
        if args.debug:
            x,y,w,h=b
            cv2.rectangle(dbg,(x,y),(x+w,y+h),(0,0,255),2)
    if args.debug:
        cv2.imwrite(str(base/"_debug_boxes.png"), dbg)
    return len(boxes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o","--output", default="dataset_panels")
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--min-area", type=int, default=25000)
    ap.add_argument("--square", action="store_true")
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--debug", action="store_true")
    # blancos/gutters
    ap.add_argument("--white-s-max", type=int, default=40, help="S máx (HSV) para blanco")
    ap.add_argument("--white-v-min", type=int, default=180, help="V mín (HSV) para blanco")
    ap.add_argument("--gutter-dilate", type=int, default=6, help="Engrosado del gutter")
    ap.add_argument("--merge-tol", type=int, default=12, help="Fusión de líneas Hough")
    # inpaint
    ap.add_argument("--inpaint", action="store_true")
    ap.add_argument("--inpaint-strength", type=int, default=3)
    ap.add_argument("--balloon-min-rel", type=float, default=0.01)
    ap.add_argument("--balloon-max-rel", type=float, default=0.32)
    args = ap.parse_args()

    p = Path(args.input)
    imgs = sorted([*p.glob("*.jpg"), *p.glob("*.png"), *p.glob("*.jpeg")]) if p.is_dir() else [p]
    total=0
    for f in imgs:
        try: total += process_one(f, args)
        except Exception as e: print(f"[WARN] {f}: {e}")
    print(f"[OK] {len(imgs)} plancha(s) | {total} viñetas → {args.output}")

if __name__ == "__main__":
    main()
