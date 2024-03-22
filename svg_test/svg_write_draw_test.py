import svgwrite

dwg = svgwrite.Drawing(filename='tmp.svg')

dwg.add(dwg.path(d=f'M 10 80 Q 95 10 180 80', stroke='black', fill='none'))

dwg.save()
