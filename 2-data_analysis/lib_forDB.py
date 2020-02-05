""" READ ME!
This script created for
1-Read all data from a sqlite db
2-create a basic nlp func for text detection from input
3-create a basic nlp func with another way for text detection from input """
import sqlite3,re
# cinection to db
con = sqlite3.connect('test.db')
# define a cursor object for run sqlite3 codes in python
cursor = con.cursor()
 # obj for get table from db
obj = cursor.execute('SELECT name from sqlite_master where type= "table"')

# get all data from db
def get_data():
	# loop for tables in db
	for i in obj:
		cursor.execute("""SELECT * FROM %s""" % (i))
		# get all data in table
		rows = cursor.fetchall()
	# retun all data like a list
	return rows
# nlp script 
def create_nlp():
	spam,not_spam,notr = [],[],[]
	for row in get_data():
		# remove newline character and leading spaces
		l1 = row[3].strip()
		# convert to lowercase to avoid mismatch
		l1 = l1.lower()
		# split comment into the words(list)
		words = re.findall(r"[\w]+",l1)
		if row[4] == 1.0:
			for word in  words:
				# crate a unique spam class
				if word not in spam and word not in not_spam:
					spam.append(word)
				# crate a unique notr class
				elif word not in notr:
					notr.append(word)
		else:
			for word in words:
				# crate a unique not_spam class
				if word not in spam and word not in not_spam:
					not_spam.append(word)
				# crate a unique notr class
				elif word not in notr:
					notr.append(word)
	text = input('1pls input a text :')
	# remove newline character and leading spaces
	l1 = text.strip()
	# convert to lowercase to avoid mismatch
	l1 = l1.lower()
	# split comment into the words(list) by everything(' ',',','.'...)
	words = re.findall(r"[\w]+",l1)
	c1 = c2 = 0
	# find the number of spam and not_spam
	for word in words:
		if word in spam:
			c1 = c1 + 1
		elif word in not_spam:
			c2 = c2 + 1
	# whichever is greater, text is closer to that class
	# and calculate accuracy c1orc2/len(text)
	if c1 > c2 :
		print('this text ',round(c1/len(words)*100,2), 'percent spam')
	elif c2 > c1:
		print('this text ',round(c2/len(words)*100,2), 'percent not spam')
	else:
		print('this text is notr')


# nlp script option2
def nlp_option2():
	positive,negative = [],[]
	for row in get_data():
		l1 = row[3].strip()
		l1.lower()
		words = re.findall(r"[\w]+",l1)
		for word in words:
			# create a spam class
			if row[4] == 1.0:
				if word not in negative:
					negative.append(word)
			# create a not_spam class
			else:
				if word not in positive:
					positive.append(word)
	text = input('2pls input a text :')
	# remove newline character and leading spaces
	l1 = text.strip()
	# convert to lowercase to avoid mismatch
	l1 = l1.lower()
	# split comment into the words(list) by everything(' ',',','.'...)
	words = re.findall(r"[\w]+",l1)
	c1 = c2 = 0
	# find the number of spam and not_spam
	for word in words:
		if word in positive:
			c1 = c1 + 1
		if word in negative:
			c2 = c2 + 1
	# whichever is greater, text is closer to that class
	# and calculate accuracy c1orc2/len(text)
	if c1 >= c2:
		print('this text ', round(c1/len(words)*100,2),'percent not spam')
	elif c2>c1:
		print('this text ', round(c2/len(words)*100,2), 'percent spam')

create_nlp()
nlp_option2()

con.close()
cursor.close()