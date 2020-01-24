# get a string
string = input('pls input a string :')

# option1
# def convert(str):
# 	print('uppercase format :', string.upper())
# 	print('lowercase format :', string.lower())
# 	print('title format : ', string.title())
# convert(string)

# option2 ascii codes
from string import ascii_lowercase, ascii_uppercase
option = int(input('input a option, for upper 1, for lower 2, for title 3 :'))
if  option == 1:
	d = dict(zip(ascii_lowercase, ascii_uppercase))
	res = ''.join([d.get(i, i) for i in string])
	print(res)
elif option == 2:
	d = dict(zip(ascii_uppercase, ascii_lowercase))
	res = ''.join([d.get(i, i) for i in string])
	print(res)
elif option == 3:
	string_splt = string.split()
	for k in string_splt:
		d = dict(zip(ascii_lowercase, ascii_uppercase))
		res = ''.join([d.get(i, i) for i in k[0]])
		print(k.replace(k[0],res), end=' ')
