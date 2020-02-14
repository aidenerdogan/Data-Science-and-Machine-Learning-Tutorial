# import numpy
import numpy as np

# S1-Arrays
# create a data
# data = [[1,2,3],[4,5,6],[7,8,9]]
# data convert to array
# array = np.array(data)
# print(array)
# print(array[0])
# print(array[1][2])

"""We'll lean numpy functions but before it I'll show you numpy power.
just check this code 'dir(np)' or 'len(dir(np))'
when i checked it has 624 functions.
this is power."""

# S2-Functions --> F1-Arange: createing an array for us.
# print(np.arange(0,100)) # 0 to 100 create a array 
# print(np.arange(0,100,5)) #0 to 100 but five and fiive create a array

# F2-zeros, eye, one functions: 
# * zeros: Creating an array consisting of 0 elements.
# print(np.zeros(100)) #100 pieces with 0 elements
# print(np.zeros((25,5))) #25 5-element arrays with elements 0
# * ones: Creating an array consisting of 1 elements.
# print(np.ones(100))
# print(np.ones((25,5)))
# * eye: watch and learn :)
# print(np.eye(20))

# F3-Linspace
# print(np.linspace(0,100,3))

# F4- Random Functions
# print(len(dir(np.random)))
# I/O:84
# *randint
# print(np.random.randint(0,100)) #or print(np.random.randint(100)) both of are same
# print(np.random.randint(0,100,5))
# *rand
# print(np.random.rand(5)) # between 0,1
# print(np.random.randn(5)) #between -1,1
# *arange
# data = np.arange(225)
# print(data)
# print(data.reshape(15,15)) #15 and 15
# print(data.cumsum()) # cumulative sum
# print(data.min())
# print(data.max())
# print(data.sum())
# *argmax(): index of max element in array
# print(data.argmax())
# *argmin(): index of min element in array
# print(data.argmin())
# *Determinant
# data1 = np.array([[1,2],[3,4]])
# print(np.linalg.det(data1)) 
# print(np.std(data))
# print(np.var(data))
# data = np.arange(0,100)
# print(data[:6]) #or daya[0:6] both of are same: 0 to 6 elements
# print(data[::3]) # 3 and 3
# data[:2] = 10
# print(data)
# print(data>50)
# array1 = np.array([5,10,15,20,25,30])
# array2 = np.array([1,2,3,4,5,6])
# note: for aritmatich operations they must be same size
# print(array1+array2)
# print(array1-array2)
# print(array1/array2) #//,%...
# print(array1*array2)
# print(array1**array2)
# print(np.sqrt(array1))
# *Tanspose
# for array1 it imposible bcs it has 1 dimension
# print(array1.shape)
# array1 = np.array([[5,10,15,20,25,30]])
# print(array1.shape)
# print(array1.T)

# for more information visit numpy oficial tutorial
# https://numpy.org/devdocs/user/quickstart.html