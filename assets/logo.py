# Generate a simple PNG logo using PIL
from PIL import Image, ImageDraw, ImageFont
import os

img = Image.new('RGBA', (300, 80), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Background
draw.rounded_rectangle([0, 0, 299, 79], radius=12, fill=(26, 127, 55, 255))

# Text
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    font = ImageFont.load_default()
    small_font = font

draw.text((15, 12), "🌱 AI Soil Advisor", font=font, fill=(255, 255, 255, 255))
draw.text((15, 52), "Empowering Indian Farmers", font=small_font, fill=(187, 247, 208, 255))

img.save(os.path.join(os.path.dirname(__file__), 'logo.png'))
print("Logo created!")
