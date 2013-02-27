import numpy as np
from time import time
a = np.random.randint(2,size=(1000,1000))
b = np.random.randint(2,size=(1000,1000))

c = a - b
d = c[c!=0]
print d

def testRuntime(f, is2d):
    arraySizes = [2**x for x in range(20,25)]
    print arraySizes
    if is2d:    
        arrays = map(lambda x: np.random.randint(5, size=(np.sqrt(x), np.sqrt(x))), arraySizes)
    else:
        arrays = map(lambda x: np.random.randint(5, size=x), arraySizes)
        
    for l in range(len(arrays)):
        start = time()
        f(arrays[l])
        end = time()
        print "%20d%20f" % (arraySizes[l], end-start)


#digitize        
testRuntime(lambda x : np.digitize(x, np.array([0,1,2,3,4,5,6])), False)

#indexing
print "Testing Indexing 2d"
testRuntime(lambda x : x[x == 1], True)

print "Testing Indexing 1d"
testRuntime(lambda x : x[x == 1], False)

