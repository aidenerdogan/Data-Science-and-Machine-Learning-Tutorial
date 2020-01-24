# get a string
string = input('pls input a string :')

# get start number
start = int(input('pls input start number :'))
# get end number
end =  int(input('pls input end number :'))

# option1
# print(string[start-1:end]) 

# option2 use slice
# slice_obj = slice(start-1,end,1)
# print(string[slice_obj])

# option3 convert to list
list1 = list(string[start-1:end])
for i in list1:
	print(i, end='')
# print(list1[start-1:end])