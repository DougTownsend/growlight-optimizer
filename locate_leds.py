import os


for ledfilename in os.listdir('./20x20/400led_4in'):
    if 'csv' not in ledfilename:
        continue
    f = open(f"{ledfilename[:-4]}.txt", "w+")
    ledfile = open(f"./20x20/400led_4in/{ledfilename}", "r")
    layer = ''
    if 'red' in ledfilename:
        layer = 'B.Fab'
    else:
        layer = 'F.Fab'
    infile_lines = ledfile.readlines()
    print(infile_lines)
    ledfile.close()
    for line in infile_lines:
        coords = line[:-1].split(',')
        coords = (round(float(coords[0])*25.4, 3), round(float(coords[1])*25.4, 3))

        f.write(f"  (gr_circle (center {coords[0]} {coords[1]}) (end {coords[0]+3.175} {coords[1]}) (layer \"{layer}\") (width 0.1) (fill none))\n")

    f.close() 