import csv
# importing required modules 
from zipfile import ZipFile as zf

# specifying the zip file name 
file_path = 'YouTube-Spam-Collection-v1.zip'

# read and  extract file from zip file exercise
# try:
# 	with zf(file_path,'r') as zip_file:
# 		# printing all the contents of the zip file 
# 		zip_file.printdir()
# 		print(type(zip_file))
# 		# for i in zip_file.extractall():
# 		# 	l1.append(i)
# 	# 	for i in list(zip_file):
# 	# 		l1.append(i)
# 	# print(l1)
# except FileExistsError  as e:
# 	print(e)

def get_file():
	l1 = []
	count = 0
	# opening the zip file in READ mode 
	with zf(file_path,'r') as zip_file:
		for i in zip_file.namelist():
			if count>0:
				l1.append(i)
			count +=1
	return l1
for i in get_file():
	print(i)
def get_data(file):
	l1 = []
	# read data from csv file
	with open(file,'r') as csv_file:
		csv_reader = csv.reader(csv_file)
		for i in csv_reader:
			l1.append(i)
	return l1[:2]

def insert_data(file_name):
	# insert data to csv file
	with open(file_name,'w') as csv_file:
		csv_writer = csv.writer(csv_file)
		# insert data to csv file from csv files in zip file
		for i in get_file():
			csv_writer.writerows(get_data(i))

# insert_data('test5.csv')
for i in get_file():
	print(get_data(i))

# https://www.google.com/search?q=extract+zip+file+python&oq=extract+xip+file+pythn&aqs=chrome.1.69i57j0l7.9183j0j7&sourceid=chrome&ie=UTF-8

# https://www.pythonforthelab.com/blog/storing-data-with-sqlite/
# print(get_data('test1.csv'))
# Task:
#      create multiple csv files with some data and compress it in zip format
#      using python script first extract it after that read files one by one and create one table using sqlite3 and based on csv files insert data in tables

