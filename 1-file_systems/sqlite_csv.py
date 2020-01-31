import csv,sqlite3,os
from zipfile import ZipFile as zf

# create table and insert table
def insert_table(table,data):
	# execute: for run the sqlite codes in python script.
	# and this line for create table in db
	c.execute("""CREATE TABLE IF NOT EXISTS %s (COMMENT_ID text,
			AUTHOR text ,
			DATE real,
			CONTENT text,
			CLASS real)""" % (table))
	# this line for insert many data to table like a list.
	c.executemany("""INSERT INTO %s (COMMENT_ID,AUTHOR,DATE,CONTENT,CLASS) VALUES (?,?,?,?,?)""" % (table),(data))
	# save changes
	conn.commit()

# insert_table('test1')
# def insert_into(data)

# get data from any file
def get_data(file):
	lst = []
	# open file like reading with 'r'
	with open(file,'r+') as f:
		# creating a csv dict reader object
		reader = csv.reader(f)
		for i in reader:
			# add to list (bcs you can't use outside dict reader object)
			lst.append(i)
	return lst



# get files from extacted zip file and get file names for table names
def get_files(file_path):
	# get your directory
	my_directory = os.getcwd()
	with zf(file_path,'r') as zip_files:
		zip_files.extractall()
		entries = os.listdir(my_directory+'/'+(file_path.split('.')[0]))
		for i in entries:
			# sqlite is not allow '-' this character for tablenames.
			tables.append(i.split('-')[0])
			# add file directory to list
			files.append(my_directory+'/'+file_path.split('.')[0]+'/'+str(i))
	return tables,files

# main func
def main():
	get_files(file_path)
	for i,j in zip(tables,files):
		insert_table(i,get_data(j))

	# read all datas from db
	c.execute("""SELECT * FROM %s""" % (i))
	rows = c.fetchall()
	for row in rows:
		print(row)


if __name__=="__main__":

	"""If database(db) name already exists it will open db or it will create and open db."""
	db_name= input('pls input a database name :')
	#connect to db
	conn = sqlite3.connect(db_name+'.db')
	# The cursor is going to allow you to execute the SQL code in python.
	c = conn.cursor()
	# print(my_directory)
	file_path = 'YouTube-Spam-Collection-v1.zip'
	tables = []
	files = []
	main()
	# close cursor
	c.close()
	# close connection from db
	conn.close()