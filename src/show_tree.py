import sys
from pathlib import Path
from collections import Counter

root = Path(sys.argv[1])
for d in sorted([root] + [p for p in root.rglob("*") if p.is_dir()]):
    files = [f for f in d.iterdir() if f.is_file()]
    exts = Counter(f.suffix.lower() for f in files)
    depth = len(d.relative_to(root).parts)
    print("  " * depth + f"{d.name}/  files={len(files)} {dict(exts)}")
print("\nNon-image files at top level:",
      [f.name for f in root.iterdir() if f.is_file()][:20])