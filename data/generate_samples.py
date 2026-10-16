"""
Sample image generator for Smart Coin Detection System.
Generates controlled, scientifically documented test images for validation and demonstration.
"""

from pathlib import Path
import numpy as np
import cv2


def draw_realistic_coin(
    canvas: np.ndarray,
    center: tuple,
    radius: int,
    color_class: str = "silver",
    scratches: bool = False,
    spots: bool = False
):
    """Draws a synthetic coin with rim shading, relief profile, and optional surface defects."""
    cx, cy = center

    # Outer soft drop-shadow
    cv2.circle(canvas, (cx + 4, cy + 4), radius + 2, (30, 30, 30), -1)

    # Base metal color
    if color_class == "copper":
        base_color = (45, 95, 185)       # BGR: reddish-copper
        rim_color = (30, 75, 155)
        core_color = (60, 115, 210)
    elif color_class == "gold":
        base_color = (50, 185, 225)      # BGR: brass-gold
        rim_color = (35, 150, 190)
        core_color = (70, 205, 245)
    elif color_class == "bimetallic":
        base_color = (50, 185, 225)      # Outer brass ring
        rim_color = (35, 150, 190)
        core_color = (195, 195, 200)     # Inner cupronickel core
    else:  # silver/nickel
        base_color = (190, 190, 195)     # BGR: silvery-gray
        rim_color = (150, 150, 155)
        core_color = (215, 215, 220)

    # Draw outer rim
    cv2.circle(canvas, (cx, cy), radius, rim_color, -1)
    cv2.circle(canvas, (cx, cy), radius - 3, base_color, -1)
    
    # Inner relief / core
    if color_class == "bimetallic":
        cv2.circle(canvas, (cx, cy), int(radius * 0.65), core_color, -1)
        cv2.circle(canvas, (cx, cy), int(radius * 0.65), rim_color, 2)
    else:
        cv2.circle(canvas, (cx, cy), int(radius * 0.88), rim_color, 1)
        cv2.circle(canvas, (cx, cy), int(radius * 0.85), core_color, -1)

    # Simulated center embossing (profile / stars / letters)
    cv2.circle(canvas, (cx, cy), int(radius * 0.45), rim_color, 1)
    cv2.putText(
        canvas,
        "LIBERTY",
        (cx - int(radius * 0.45), cy + int(radius * 0.12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        radius / 150.0,
        rim_color,
        1,
        cv2.LINE_AA
    )

    # Optional defects
    if scratches:
        # High contrast scratch lines
        cv2.line(canvas, (cx - radius // 2, cy - radius // 3), (cx + radius // 3, cy + radius // 4), (20, 20, 25), 2)
        cv2.line(canvas, (cx - radius // 4, cy + radius // 5), (cx + radius // 2, cy - radius // 5), (30, 30, 35), 1)

    if spots:
        # Dark oxidation spots / corrosion flecks
        cv2.circle(canvas, (cx - radius // 3, cy - radius // 4), 6, (15, 15, 20), -1)
        cv2.circle(canvas, (cx + radius // 4, cy + radius // 3), 8, (25, 45, 20), -1)
        cv2.circle(canvas, (cx, cy - radius // 2), 5, (20, 25, 20), -1)


def generate_all_samples(output_dir: Path):
    """Generates all documented sample test images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    w, h = 800, 600

    def make_background():
        # Soft neutral gray background simulating light studio paper table
        bg = np.full((h, w, 3), 235, dtype=np.uint8)
        # Subtle texture noise
        noise = np.random.normal(0, 3, (h, w, 3)).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return bg

    # 1. Single Coin
    img1 = make_background()
    draw_realistic_coin(img1, (400, 300), radius=90, color_class="silver")
    cv2.imwrite(str(output_dir / "sample_1_coin.jpg"), img1)

    # 2. Three Separated Coins
    img3 = make_background()
    draw_realistic_coin(img3, (200, 300), radius=70, color_class="copper")  # Cent
    draw_realistic_coin(img3, (400, 300), radius=80, color_class="silver")  # Nickel
    draw_realistic_coin(img3, (600, 300), radius=92, color_class="silver")  # Quarter
    cv2.imwrite(str(output_dir / "sample_3_coins.jpg"), img3)

    # 3. Five Separated Coins
    img5 = make_background()
    draw_realistic_coin(img5, (160, 220), radius=68, color_class="copper")  # Penny
    draw_realistic_coin(img5, (380, 200), radius=78, color_class="silver")  # Nickel
    draw_realistic_coin(img5, (620, 220), radius=65, color_class="silver")  # Dime
    draw_realistic_coin(img5, (260, 420), radius=88, color_class="silver")  # Quarter
    draw_realistic_coin(img5, (520, 420), radius=110, color_class="silver") # Half Dollar
    cv2.imwrite(str(output_dir / "sample_5_coins.jpg"), img5)

    # 4. Touching Coins
    img_touch = make_background()
    # Coins touching at boundary: center distance = r1 + r2 = 80 + 80 = 160
    draw_realistic_coin(img_touch, (320, 300), radius=80, color_class="silver")
    draw_realistic_coin(img_touch, (480, 300), radius=80, color_class="gold")
    cv2.imwrite(str(output_dir / "sample_touching_coins.jpg"), img_touch)

    # 5. Overlapping Coins
    img_overlap = make_background()
    # Bottom coin
    draw_realistic_coin(img_overlap, (340, 300), radius=85, color_class="silver")
    # Overlapping top coin
    draw_realistic_coin(img_overlap, (450, 300), radius=85, color_class="gold")
    cv2.imwrite(str(output_dir / "sample_overlapping_coins.jpg"), img_overlap)

    # 6. No Coins (Empty tabletop)
    img_empty = make_background()
    cv2.imwrite(str(output_dir / "sample_no_coins.jpg"), img_empty)

    # 7. Blurry Coin
    img_blur = make_background()
    draw_realistic_coin(img_blur, (400, 300), radius=85, color_class="silver")
    img_blur = cv2.GaussianBlur(img_blur, (45, 45), sigmaX=15)
    cv2.imwrite(str(output_dir / "sample_blurry_coin.jpg"), img_blur)

    # 8. Low-light Coin
    img_dark = make_background()
    draw_realistic_coin(img_dark, (400, 300), radius=85, color_class="silver")
    img_dark = (img_dark.astype(np.float32) * 0.22).astype(np.uint8)
    cv2.imwrite(str(output_dir / "sample_low_light.jpg"), img_dark)

    # 9. Damaged / Anomaly Coin
    img_dmg = make_background()
    draw_realistic_coin(img_dmg, (400, 300), radius=90, color_class="gold", scratches=True, spots=True)
    cv2.imwrite(str(output_dir / "sample_damaged_coin.jpg"), img_dmg)

    print(f"Generated 9 test images in {output_dir}")


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "sample_images"
    generate_all_samples(out_path)
