from PIL import Image

# img = Image.open("jaxcont_logo.png").convert("RGBA")
# data = img.getdata()

# new_data = []
# threshold = 200  # هرچه بیشتر، سفیدهای بیشتری حذف می‌شوند

# for r, g, b, a in data:
#     if r > threshold and g > threshold and b > threshold:
#         new_data.append((255, 255, 255, 0))  # transparent
#     else:
#         new_data.append((r, g, b, a))

# img.putdata(new_data)
# img.save("jaxcont_logo_transparent.png")


from PIL import Image

img = Image.open("jaxcont_logo.png")

new_size = (
    img.width // 3,
    img.height // 3,
)

img = img.resize(new_size, Image.Resampling.LANCZOS)
img.save("jaxcont_logo_small.png")