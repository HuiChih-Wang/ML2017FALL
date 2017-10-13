from PIL import Image
import sys
import numpy as np

# read image
img = Image.open(sys.argv[1])
img = np.asarray(img)

img_darken = Image.fromarray(img/2)

save_name = "Q2.jpg"
img_darken.save(save_name)