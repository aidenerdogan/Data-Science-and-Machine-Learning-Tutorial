
# get a string
string = input('pls input a string :')

#get a count
count = int(input('pls input a count :'))


#option1 for loop
# for i in range(count):
# 	print(string, end='')
# print('\n')

# option2 recursion func.
# import sys 
# sys.setrecursionlimit(10**6) 
# def write(str, x):
# 	if x>0: 
# 		print(string, end='')
# 	x -= 1
# 	return  write(string,x)
# write(string,count)

#option3 while loop
while (count>0):
	print(string, end='')
	count -=1