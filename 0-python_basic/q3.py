# get month number
number = int(input('pls input a month number between 1 to 12 :'))
 
# option1
def find(x):
	if x>12 or x<1:
		print('pls input between 1 to 12')
	else:
		quarter = (x-1)//3+1
		sequence = 1
		if x%3 == 0:
			sequence = 3
		else:
			sequence = x%3
		print('quarter: ', quarter,'and sequence: ',sequence)
find(number)

# option1.5
 # a note: i can use condition for 12 months but it is not a solution :)

 