"""Render a simple synthetic start frame: red ball on a beige table.

832x480 matches HunyuanVideo-1.5 480p I2V native size so the server
won't have to resize our conditioning image.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

W, H = 832, 480
OUT = Path(__file__).parent / "assets" / "start_frame.png"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (W, H), (214, 191, 153))  # warm beige "table"
    draw = ImageDraw.Draw(img)

    # horizon band: darker wall behind the table
    draw.rectangle([0, 0, W, int(H * 0.45)], fill=(170, 155, 130))

    # ball: solid red circle, slightly left of center, sitting "on" the table
    cx, cy, r = int(W * 0.42), int(H * 0.62), 60
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 40, 40))

    # soft shadow ellipse under the ball
    shadow = Image.new("L", (W, H), 0)
    sdraw = ImageDraw.Draw(shadow)
    sdraw.ellipse([cx - r - 10, cy + r - 10, cx + r + 10, cy + r + 20], fill=120)
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    img.paste((0, 0, 0), mask=shadow)

    img.save(OUT)
    print(f"wrote {OUT} ({W}x{H})")


if __name__ == "__main__":
    main()
