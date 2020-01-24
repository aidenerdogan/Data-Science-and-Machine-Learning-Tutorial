#creating a new lists
lis1 = []
lis2 = []

#get lists
list1 = input('pls input a list for list1 :')
list2 = input('pls input a list for list2 :')	


#compare lists

#option1 lists same or not
# if list1 == list2:
# 	print('list1 is subset of list2.')
# else:
# 	print('list1 is not subset of list2.')

#option2 lists same or not with all lib

# if len(list1)>=len(list2):
# 	if(all(i in list1 for i in list2)):
# 		print('list2 is subset of list1.')
# 	else:
# 		print('list2 is not subset of list1.')
# else:
# 	if(all(i in list2 for i in list1)):
# 		print('list1 is subset of list2.')
# 	else:
# 		print('list1 is not subset of list2.')

#option3 check susbset with subset lib

# if len(list1)>=len(list2):
# 	if(set(list2).issubset(set(list1))):
# 		print('list2 is subset of list1.')
# 	else:
# 		print('list2 is not subset of list1.')
# else:
# 	if(set(list1).issubset(set(list2))):
# 		print('list1 is subset of list2.')
# 	else:
# 		print('list1 is not subset of list2.')

#option4 use flag 
def check_subset(a,b):
	flag = True
	if len(list1)>=len(list2):
		for i in list2:
			if i not in list1:
				flag = False
		if flag == True:
			print('list2 is subset of list1.')
		else:
			print('list2 is not subset of list1.')
	else:
		for i in list1:
			if i not in list2:
				flag = False
		if flag == True:
			print('list1 is subset of list2.')
		else:
			print('list1 is not subset of list2.')

check_subset(list1,list2)

