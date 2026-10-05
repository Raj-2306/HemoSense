import sys
from pathlib import Path
from PIL import Image, ImageDraw

cls = sys.argv[1]
files = sorted(Path("data/unique", cls).glob("*.jpg"))[:3]
W = 1200
rows = []
for f in files:
    im = Image.open(f).convert("RGB")
    im = im.resize((W, int(im.height * W / im.width)))
    d = ImageDraw.Draw(im)
    for i in range(1, 4):                      # quarter lines
        d.line([(W * i // 4, 0), (W * i // 4, im.height)], fill=(0, 0, 255), width=2)
    d.text((8, 8), f.name, fill=(0, 0, 255))
    rows.append(im)
sheet = Image.new("RGB", (W, sum(r.height for r in rows)), "white")
y = 0
for r in rows:
    sheet.paste(r, (0, y)); y += r.height
out = Path(f"results/dataset_report/layout_{cls.replace(' ', '_')}.png")
sheet.save(out)
print(out)