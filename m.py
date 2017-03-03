import numpy
a=numpy.array([[1,2],
              [3,0]
              ])

b=numpy.array([[1,1],[0,0]]) 
c=(a*b)
fimg=numpy.array([0])
for k in xrange(2) :
 	for l in xrange(2) :
 		fimg+=c[k,l]
print(fimg)