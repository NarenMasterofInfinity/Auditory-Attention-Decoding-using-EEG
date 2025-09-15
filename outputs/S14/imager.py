from PIL import Image, ImageDraw, ImageFont

# Load the uploaded image
img_path = "A_edges_delta.png"
img = Image.open(img_path).convert("RGBA")

# Create a drawing context
draw = ImageDraw.Draw(img)

# Define font (fallback to default if DejaVuSans not available)
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
except:
    font = ImageFont.load_default()

# Example node coordinates (manual approximations since we don’t know exact layout)
# Format: (x, y, "label", "meaning")
annotations = [
    (250, 100, "Fz", "Frontal → attention control"),
    (150, 200, "T7", "Left Temporal → auditory cortex"),
    (350, 200, "T8", "Right Temporal → auditory cortex"),
    (250, 300, "Pz", "Parietal → integration"),
    (250, 200, "Cz", "Central → sensory integration"),
]

# Draw circles + labels
for (x, y, node, meaning) in annotations:
    r = 20
    draw.ellipse((x-r, y-r, x+r, y+r), outline="red", width=3)
    draw.text((x+25, y-10), f"{node}: {meaning}", fill="black", font=font)

# Save annotated image
out_path = "annotated_connections.png"
img.save(out_path)