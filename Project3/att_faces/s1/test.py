import numpy as np

from sklearn.decomposition import PCA
from PIL import Image
im = Image.open('1.pgm')
print im.format, im.size, im.mode
im.show()
# a = np.array(im)
# data = np.array(a.reshape(1,10304))

# for num in range(2,6):
# 	im = Image.open(str(num)+".pgm")
# 	a = np.array(im)
	
# 	data = np.concatenate((data,a.reshape(1,10304)))


# print data.shape



# pca = PCA(n_components=2)
# X_r = pca.fit(data).transform(data)
# print X_r.shape

	# print im.format, im.size, im.mode
	# im.show()