import cv2
import sys
import numpy as np
import scipy.misc

from PIL import Image


file_name = sys.argv[1].split('/')[-1]
img_name_text = file_name.split('.')[0]


img = cv2.imread(sys.argv[1]);

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

new_image_name = img_name_text+"_adversarial.png";

im = Image.fromarray(dst)
im.save('./adversary/'+new_image_name)

print(new_image_name);


sys.stdout.flush()


