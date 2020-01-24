# get lists
list1 = input('pls input list1 :')
list2 = input('pls input list2 :')

# option1 compare func
# def compare(l1,l2):
# 	templist = []
# 	for i in l1:
# 		if i not in l2:
# 			templist.append(i)
# 	print(templist)
# 	for j in l2:
# 		if j not in l1 and j not in templist:
# 			templist.append(j)
# 	print(templist)
# 	return templist
# print(compare(list1,list2))

# option2 use list and set 
# def compare(l1, l2):
# 	return (list(set(l1) - set(l2))) + (list(set(l2) - set(l1)))
# print(compare(list1,list2))

# # option3 only set xxx
# def compare(l1,l2):
# 	templist = []
# 	temp = set(l1) & set(l2)
# 	print(temp)
# 	for i in zip(l1,l2):
# 		if i not in temp:
# 			templist.append(i)
# 		# if j not in temp:
# 		# 	templist.append(j)
# 	print(templist)
# 	return templist
# compare(list1,list2)
# print(compare(list1,list2))

# def compare(l1,l2):
# 	templist =  [i for i, j in zip(l1, l2) if i == j]
# 	return templist
# print(compare(list1,list2))
# print([i for i, j in zip(list1, list2) if i == j])