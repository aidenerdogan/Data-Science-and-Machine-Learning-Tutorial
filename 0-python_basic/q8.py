# get a sentences
sentence = input('pls input a sentence :')

# option1
flag_list = []
for i in sentence.split():
	if i not in flag_list:
		flag_list.append(i)
		print(i,' : ', sentence.split().count(i), end=' ' )

# option2
# splt_sentence = sentence.split()
# # print(splt_sentence)
# flag_list = []
# for i in splt_sentence:` 
# 	count = 0
# 	for j in splt_sentence:
# 		if i == j:
# 			count += 1 
# 	if i not in flag_list:
# 		flag_list.append(i)
# 		print(i,' :',count, end=' ')


for i in str1.split():
...     if i not in temp:
...             temp[i] = 0
...             temp[i] = temp[i]+1
...     else:
...             temp[i] = temp[i]+1
