from pathlib import Path
from PIL import Image, ImageDraw

root, W, H = Path("data/unique"), 960, 443
classes = sorted(p for p in root.iterdir() if p.is_dir())
tiles = []
for d in classes:
    im = Image.open(sorted(d.glob("*.jpg"))[0]).convert("RGB").resize((W, H))
    ImageDraw.Draw(im).text((10, 10), d.name, fill=(0, 0, 255))
    tiles.append(im)
for i in range(2):
    sheet = Image.new("RGB", (W, H * 4), "white")
    for j, t in enumerate(tiles[i * 4:(i + 1) * 4]):
        sheet.paste(t, (0, j * H))
    out = Path(f"results/dataset_report/class_montage_{i + 1}.png")
    sheet.save(out)
    print(out, [c.name for c in classes[i * 4:(i + 1) * 4]])