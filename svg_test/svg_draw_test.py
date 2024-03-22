import drawsvg as draw

d = draw.Drawing(200, 100, origin='center')

# Draw an arbitrary path (a triangle in this case)
p = draw.Path(stroke_width=2, stroke='lime', fill='black', fill_opacity=0.2)
p.M(-10, -60)  # Start path at point (-10, -20)
p.C(30, 10, 30, -50, 70, -20)  # Draw a curve to (70, -20)
d.append(p)

d.save_svg(fname='tmp1.svg')

"""
png_io = BytesIO()
d.save_png(png_io)
png_io.seek(0)

png_bytes = np.asarray(bytearray(png_io.read()), dtype=np.uint8)
image = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""