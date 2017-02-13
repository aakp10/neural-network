import numpy as np
  
def softmax(x):

 	e_x=np.exp(x)

 	return e_x/e_x.sum()

ip2=np.random.rand(2,4)             #array for set of all ip ele with a column containing one ip set


sfmx=np.random.rand(2,4)             #array to hold softmax result

score=np.random.rand(2,4)              #array to hold input going to activ func ===matrix mult with wt. +bias
ip1=np.array([1,1,0])                    #individual ip

ip2=np.array([1,0,0])

ip3=np.array([1,1,1])

ip4=np.array([1,0,1])

input=np.column_stack((ip1,ip2))         #combining ip under 1 array by column

input=np.column_stack((input,ip3))

input=np.column_stack((input,ip4))

desire=np.array([0,0,1,0
                          ])
w=np.random.rand(1,3)*0.01

print ' weights \n\n'

print w  
for i in xrange(4) :
	
    score[:,i]=np.dot(w,input[:,i])
  

ip2=np.maximum(0,score)

#for i in xrange(4) :
	
#sfmx[:,0]=softmax(ip2[:,0])    # sending arg to softmax func 1 column at atime i.e one input data set
n=5
e=desire-ip2
delta=e*1  #since using relu

for j in xrange(4) :

  	wchange[:,]=delta*input
w+=wchange 








print 'this is input\n\n'

print input

print ' weights \n\n'

print w


print 'input into activation function\n\n'

print score

print 'output of Re lu\n\n'

print(ip2)

print 'output of error\n\n '
print e
print 'desired\n'
print desire
print 'delta\n'
print delta
print'w change'
print wchange