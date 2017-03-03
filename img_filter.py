import scipy
import numpy
from scipy.misc import imread ,imshow
from scipy import signal
img=scipy.misc.imread("/home/aakp/Pictures/krishna.jpg", flatten=1)
#resized img
reimg=scipy.misc.imresize(img,(682,682), interp='bilinear', mode=None)
#filter
filter=numpy.array([[2,4,5,4,2],
					[4,9,12,9,4],
					[5,12,15,12,5]
					,[4,9,12,9,4]
					,[2,4,5,4,2]
					])
filter.astype(float)
a=numpy.array((5,5),dtype='float')
fimg=numpy.array((678,678),dtype='float')
for i in xrange(682) :
	for j in xrange(682):
			 a=reimg[i:i+5,j:j+5]*filter
#convolving 5x5 kernel over reimg and storing in fimg
			 for k in xrange(5) :
			 	for l in xrange(5) :
			 		 		fimg[i,j]+=a[k,l]

			#fimg[i,j]=signal.convolve2d(reimg[i:i+6,j:j+6],filter, boundary='symm', mode='same
scipy.misc.imshow(fimg)
scipy.misc.imshow(reimg)