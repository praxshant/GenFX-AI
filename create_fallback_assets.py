"""
Generate the placeholder shown when every image provider fails.
Run once before launching the app: python create_fallback_assets.py
Requires Pillow (pip install Pillow).
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1024, 576


def make_gradient_image(label: str, output_path: Path) -> None:
    """Create a dark gradient placeholder image with a centered label."""
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)

    # Dark charcoal gradient: top-left #0F0F0F → bottom-right #242424
    for y in range(H):
        ratio = y / H
        r = int(0x0F + (0x24 - 0x0F) * ratio)
        g = int(0x0F + (0x24 - 0x0F) * ratio)
        b = int(0x0F + (0x24 - 0x0F) * ratio)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # Subtle grid / crosshair lines — viewfinder aesthetic
    grid_color = (42, 42, 42)
    # Rule-of-thirds grid
    for x in [W // 3, 2 * W // 3]:
        draw.line([(x, 0), (x, H)], fill=grid_color, width=1)
    for y in [H // 3, 2 * H // 3]:
        draw.line([(0, y), (W, y)], fill=grid_color, width=1)

    # Center crosshair
    cx, cy = W // 2, H // 2
    draw.line([(cx - 30, cy), (cx + 30, cy)], fill=(80, 70, 55), width=1)
    draw.line([(cx, cy - 30), (cx, cy + 30)], fill=(80, 70, 55), width=1)

    # Corner brackets
    bracket_len = 20
    bracket_color = (100, 85, 65)
    corners = [(10, 10), (W - 10, 10), (10, H - 10), (W - 10, H - 10)]
    for bx, by in corners:
        dx = 1 if bx < W // 2 else -1
        dy = 1 if by < H // 2 else -1
        draw.line([(bx, by), (bx + dx * bracket_len, by)], fill=bracket_color, width=2)
        draw.line([(bx, by), (bx, by + dy * bracket_len)], fill=bracket_color, width=2)

    # Main label (center)
    try:
        font_large = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 13)
    except IOError:
        font_large = ImageFont.load_default()
        font_small = font_large

    label_color = (232, 213, 183)      # --accent warm cream
    muted_color = (74, 72, 69)         # --text-tertiary

    # Draw main label
    bbox = draw.textbbox((0, 0), label, font=font_large)
    lw = bbox[2] - bbox[0]
    lh = bbox[3] - bbox[1]
    draw.text(((W - lw) // 2, (H - lh) // 2 - 10), label, fill=label_color, font=font_large)

    # Sub-caption
    sub = "GENFX  ·  FALLBACK ASSET"
    sbbox = draw.textbbox((0, 0), sub, font=font_small)
    sw = sbbox[2] - sbbox[0]
    draw.text(((W - sw) // 2, (H // 2) + 22), sub, fill=muted_color, font=font_small)

    # Bottom-left metadata
    meta = f"{W}×{H}  ·  PLACEHOLDER"
    draw.text((14, H - 26), meta, fill=muted_color, font=font_small)

    img.save(str(output_path), format="PNG")
    print(f"Created: {output_path}")


if __name__ == "__main__":
    make_gradient_image("FALLBACK IMAGE", ASSETS_DIR / "fallback_image.png")
