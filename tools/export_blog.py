"""Export the notes notebook to a self-contained markdown folder for a blog.

Output layout (default: dandi_scraper/blog_export/):
    index.md        markdown + code (hide_code cells show output only)
    images/         every image the post uses: linked figures, cell
                    attachments and plot outputs, with web-safe names

Usage:
    python -I tools/export_blog.py
    python -I tools/export_blog.py --out some/dir --img-prefix /assets/intra-dandi/
"""
import argparse
import base64
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "dandi_scraper" / "notes_on_intra_ephys.ipynb"
IMG_RE = re.compile(r"!\[([^\]]*)\]\(\s*(<[^>]+>|[^)\s]+)\s*\)")


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


class Images:
    def __init__(self, out_dir, prefix):
        self.dir = out_dir / "images"
        self.prefix = prefix
        self.by_source = {}  # source key -> published name
        self.used = set()

    def _claim(self, name):
        stem, dot, ext = name.rpartition(".")
        cand, i = name, 1
        while cand in self.used:
            cand = f"{stem}_{i}{dot}{ext}"
            i += 1
        self.used.add(cand)
        return cand

    def add_bytes(self, key, name, data):
        if key not in self.by_source:
            name = self._claim(safe_name(name))
            (self.dir / name).write_bytes(data)
            self.by_source[key] = name
        return self.prefix + self.by_source[key]

    def add_file(self, path):
        key = str(path.resolve())
        if key not in self.by_source:
            name = self._claim(safe_name(path.name))
            shutil.copyfile(path, self.dir / name)
            self.by_source[key] = name
        return self.prefix + self.by_source[key]


def rewrite_images(src, cell, images):
    attachments = cell.get("attachments", {})

    def repl(m):
        alt, target = m.group(1), m.group(2).strip("<>")
        if alt.lower() in ("alt text", "image.png") or alt.endswith(".png"):
            alt = ""
        if target.startswith("attachment:"):
            name = target[len("attachment:"):]
            mime, b64 = next(iter(attachments[name].items()))
            url = images.add_bytes(f"att:{cell['id']}:{name}",
                                   f"{cell['id']}_{name}", base64.b64decode(b64))
        elif re.match(r"https?://", target):
            return m.group(0)
        else:
            path = (NB.parent / target).resolve()
            if not path.exists():
                print(f"WARNING: missing image {target}")
                return m.group(0)
            url = images.add_file(path)
        return f"![{alt}]({url})"

    return IMG_RE.sub(repl, src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "dandi_scraper" / "blog_export"))
    ap.add_argument("--img-prefix", default="images/",
                    help="URL prefix for images in the markdown (e.g. /assets/post/)")
    ap.add_argument("--no-code", action="store_true",
                    help="drop all code, keep only figures from code cells")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    images = Images(out, args.img_prefix)

    nb = json.loads(NB.read_text())
    parts = []
    for cell in nb["cells"]:
        src = "".join(cell["source"]).strip()
        if cell["cell_type"] == "markdown":
            parts.append(rewrite_images(src, cell, images))
            continue
        if cell["cell_type"] != "code":
            continue
        hidden = "hide_code" in cell.get("metadata", {}).get("tags", []) or args.no_code
        if not hidden and src:
            parts.append(f"```python\n{src}\n```")
        for n, o in enumerate(cell.get("outputs", [])):
            data = o.get("data", {})
            if "image/png" in data:
                url = images.add_bytes(f"out:{cell['id']}:{n}", f"{cell['id']}_out{n}.png",
                                       base64.b64decode(data["image/png"]))
                parts.append(f"![]({url})")

    (out / "index.md").write_text("\n\n".join(parts) + "\n")
    print(f"wrote {out / 'index.md'} with {len(images.used)} images")


if __name__ == "__main__":
    main()
