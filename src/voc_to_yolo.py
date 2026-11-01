"""FOD-A Pascal VOC (XML) -> YOLO (.txt) format dönüştürücü."""
import shutil
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOC = ROOT / "dataset" / "FODPascalVOCFormat-V.2.1" / "VOC2007"
OUT = ROOT / "data" / "yolo"


def safe_copy(src: Path, dst: Path, tries: int = 8) -> None:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    for k in range(tries):
        try:
            shutil.copyfile(src, dst)
            return
        except PermissionError:
            time.sleep(0.5 * (k + 1))
    raise PermissionError(f"Kopyalanamadı: {src}")


def parse(xml_path: Path):
    r = ET.parse(xml_path).getroot()
    w = float(r.findtext("size/width"))
    h = float(r.findtext("size/height"))
    objs = []
    for o in r.findall("object"):
        b = o.find("bndbox")
        box = [float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        objs.append((o.findtext("name").strip(), box))
    return w, h, objs


def main() -> None:
    ids = {
        "train": (VOC / "ImageSets/Main/trainval.txt").read_text().split(),
        "val": (VOC / "ImageSets/Main/test.txt").read_text().split(),
    }
    all_ids = ids["train"] + ids["val"]

    parsed = {i: parse(VOC / "Annotations" / f"{i}.xml") for i in all_ids}
    counts = Counter(n for _, _, objs in parsed.values() for n, _ in objs)
    classes = sorted(counts)
    cidx = {c: k for k, c in enumerate(classes)}

    for split, split_ids in ids.items():
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in split_ids:
            w, h, objs = parsed[i]
            safe_copy(VOC / "JPEGImages" / f"{i}.jpg", OUT / "images" / split / f"{i}.jpg")
            lines = []
            for name, (x1, y1, x2, y2) in objs:
                x1, x2 = max(0.0, x1), min(w, x2)
                y1, y2 = max(0.0, y1), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(
                    f"{cidx[name]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                )
            (OUT / "labels" / split / f"{i}.txt").write_text("\n".join(lines), encoding="utf-8")

    names = "\n".join(f"  {k}: {c}" for k, c in enumerate(classes))
    (OUT / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/val\nnc: {len(classes)}\nnames:\n{names}\n",
        encoding="utf-8",
    )
    print(f"train={len(ids['train'])} val={len(ids['val'])} nc={len(classes)}")
    for c in classes:
        print(f"{cidx[c]:2d} {c:25s} {counts[c]}")


if __name__ == "__main__":
    main()
