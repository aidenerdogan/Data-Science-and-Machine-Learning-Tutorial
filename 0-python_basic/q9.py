# lists
l1 = [1,2,3,4,5,6,7,'a','b','c']
l2 = ['c','d','e',3,4,5,6,7,8,9]

# option1 '+'operation
# l3 = l1 + l2
# print(l3)

# optionn2 * operation
# l3 = [*l1,*l2]
# print(l3)

# option3 set operation (only unique ones)
# l3 = list(set(l1 + l2))
# print(l3)

# option4 extend
# l3 = []
# l3.extend(l1)
# l3.extend(l2)
# print(l3)

# option5 iteratools
# import itertools
# l3 = []
# for item in itertools.chain(l1, l2):
#     l3.append(item)
# print(l3)

# option5.5 similar op5 just only unique ones
# import itertools
# l3 = []
# for item in itertools.chain(l1, l2):
# 	if item not in l3:
# 		l3.append(item)
# print(l3)

# option6 yield and list
# def merge(list1, list2):
#     yield from list1
#     yield from list2
# l3 = list(merge(l1,l2))
# print(l3)

# option7 for loop
# l3 = []
# for (i,j) in zip(l1,l2):
# 	l3.append(i)
# 	l3.append(j)
# print(l3)

# option8 ?
# l3 = []
# l3.append(l1)
# l3.append(l2)
# print(l3)

# option9 with sorted but only int
# l3 = sorted(sorted(l1) + sorted(l2))
# print(l3)