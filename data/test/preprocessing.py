import os
import cv2
import numpy as np
import gif2numpy as g2n



a = os.listdir('./images')
b = os.listdir('./mask')
a.sort()
b.sort()

for (img_path, mask_path) in zip(a, b):
	print(img_path, mask_path)
	
	img = cv2.imread('./images/' + img_path)

	np_frames, extensions, image_specifications = g2n.convert('./mask/' + mask_path)
	mask = np.array(np_frames[0] / 255).astype(np.uint8)

	img = img * mask

	cv2.imwrite('./afterMask_red/' + img_path[:-4] + '.tif', img[2])
	