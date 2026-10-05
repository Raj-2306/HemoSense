import sys, shutil, hashlib
from pathlib import Path
from collections import Counter, defaultdict

SRC = Path(sys.argv[1])
OUT = Path("data/unique")
EXTS = {".jpg", ".jpeg", ".png"}

files = [p for p in SRC.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
groups = defaultdict(list)
for p in files:
    groups[hashlib.md5(p.read_bytes()).hexdigest()].append(p)

print(f"Files: {len(files)} | unique by content: {len(groups)}")

print("\nFolders above the label folder (shows where copies live):")
for k, v in Counter("/".join(p.relative_to(SRC).parts[:-2]) for p in files).most_common():
    print(f"  {v:4d}  {k or '(top level)'}")

print("\nCopies per unique image:", dict(Counter(len(v) for v in groups.values())))

big = next((v for v in groups.values() if len(v) > 1), None)
if big:
    print("\nExample duplicate group:")
    for p in big[:5]:
        print("  ", p.relative_to(SRC))

per_class = Counter()
for h, paths in groups.items():
    labels = {p.parent.name for p in paths}
    if len(labels) > 1:
        print("LABEL CONFLICT:", labels)
        continue
    label = labels.pop()
    dst = OUT / label
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths[0], dst / f"{h[:10]}{paths[0].suffix.lower()}")
    per_class[label] += 1

print("\nUnique images per class:")
for k in sorted(per_class):
    print(f"  {k:4s} {per_class[k]}")
print(f"\nCopied unique images to: {OUT.resolve()}")