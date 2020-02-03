# nlp script 
def create_nlp():
	spam = []
	not_spam = []
	notr = []
	for row in get_data():
		# remove newline character and leading spaces
		l1 = row[3].strip()
		# convert to lowercase to avoid mismatch
		l1 = l1.lower()
		# split comment into the words(list)
		words = l1.split()
		if row[4] == 1.0:
			for word in  words:
				if word not in spam and word not in not_spam:
					spam.append(word)
				elif word not in notr:
					notr.append(word)
		else:
			for word in words:
				if word not in spam and word not in not_spam:
					not_spam.append(word)
				elif word not in notr:
					notr.append(word)
	text = input('pls input a text :')
	# remove newline character and leading spaces
	l1 = text.strip()
	# convert to lowercase to avoid mismatch
	l1 = l1.lower()
	# split comment into the words(list)
	words = l1.split()
	c1 = 0
	c2 = 0
	for word in words:
		if word in spam:
			c1 = c1 + 1
		elif word in not_spam:
			c2 = c2 + 1
	if c1/len(words) > c2/len(words):
		print('this text ',round(c1/len(words)*100,2), 'percent spam')
	elif c2/len(words) > c1/len(words):
		print('this text ',round(c2/len(words)*100,2), 'percent not spam')
	else:
		print('this text is notr')
	# print(spam)
	# print(not_spam)
	# print(notr)

# create_nlp()

def nlp_option2():
	positive = []
	negative = []
	for row in get_data():
		l1 = row[3].strip()
		l1.lower()
		words = l1.split()
		for word in words:
			if row[4] == 1.0:
				if word not in negative:
					negative.append(word)
			else:
				if word not in positive:
					positive.append(word)
	text = input('pls input a text :')
	# remove newline character and leading spaces
	l1 = text.strip()
	# convert to lowercase to avoid mismatch
	l1 = l1.lower()
	# split comment into the words(list)
	words = l1.split()
	c1 = 0
	c2 = 0
	for word in words:
		if word in positive:
			c1 = c1 + 1
		elif word in negative:
			c2 = c2 + 1
	if c1 >= c2:
		print('this text ', round(c1/len(words)*100,2),'percent not spam')
	elif c2>c1:
		print('this text ', round(c2/len(words)*100,2), 'percent spam')
nlp_option2()