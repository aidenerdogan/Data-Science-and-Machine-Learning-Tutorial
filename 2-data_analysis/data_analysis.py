import sqlite3
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
# check comment for spam or not
def fetch_spam():
	# loop for all data
	for row in get_data():
		# check comment spam or not with already class column
		if row[4] == 1.0:
			print(row[0],'Spam')
		else:
			print(row[0],'Not Spam')
# fetc_spam()
# get repeadedly
def get_repeat():
	# Create an empty dictionary 
	d = dict()
	# loop i each comment
	for row in get_data():
		# remove newline character and leading spaces
		l1 = row[3].strip()
		# convert to lowercase to avoid mismatch
		l1 = l1.lower()
		# split comment into the words(list)
		words = l1.split()
		# loop for words
		for word in words:
			# check word already in dict or not
			if word in d:
				# Increment count of word by 1
				d[word] = d[word] + 1
			else:
				d[word] = 1
	# print the dict contents
	for key in (d.keys()):
		print(key, ":", d[key])

# get_repeat()

# get author repeatedly
def get_author_repeat():
	d = dict()
	# loop in all data
	for row in get_data():
		# row[1] for authors
		# and chech author in dict or not
		if row[1] in d:
			# if author already in dict, increment count by 1
			d[row[1]] = d[row[1]] + 1
		else:
			d[row[1]] = 1
	# print dict contents
	for key in (d.keys()):
		if d[key] > 1:
			print(key, ":", d[key])
# get_author_repeat()

 # get author spam repeatedly
def get_author_spam_repeat():
	d = dict()
	d2 = dict()
	# loop in all data
	for row in get_data():
		# row[4] for spam calass
		# check spam for author comments
		if row[4] == 1.0:
			# and chech author in dict or not
			if row[1] in d:
				# if author already in dict, increment count by 1
				d[row[1]] = d[row[1]] + 1
			else:
				d[row[1]] = 1
		# check not spam for author comments
		else:
			if row[1] in d2:
				d2[row[1]] = d2[row[1]]
			else:
				d2[row[1]] = 1

	for dic1,dic2 in zip(d.keys(),d2.keys()):
		print(dic1, ":", d[dic1], 'spam')
		print(dic2, ":", d2[dic2], 'not spam')
	# print dict contents
	# for key in (d.keys()):
	# 	if d[key] > 1:
	# 		print(key, ":", d[key])
	# for key in (d2.keys()):
	# 	print(key, ":", d2[key])
# get_author_spam_repeat()
