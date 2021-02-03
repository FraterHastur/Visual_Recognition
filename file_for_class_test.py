import image_tools
import numpy as np
from pylab import *

image = image_tools.ImageHistogram(file_path='ObjTest_good.jpg')
image.hist_equation()

