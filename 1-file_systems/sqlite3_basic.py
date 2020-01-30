"""This codes for basic sqlite exircise. 
Pls run step by step."""
# import sqlite3
import sqlite3 as s3

'''It does not exist connect will create a data base (db) or 
It does exist connect will open db.'''
# Step 1
# create/open a db
conn = s3.connect('tests.db')
# Step 2
# The cursor is going to allow you to execute the SQL code in python.
# c = conn.cursor()
# create columns for table
c.execute('''CREATE TABLE test (
	name text,
	surname text,
 	mail text)''')
#  Step 3
# insert values to table
# c.execute("INSERT INTO test VALUES ('ahmet','erdogan','ahmet.com')")
# Step 4
# read table by mail
# c.execute("SELECT * FROM test WHERE mail='ahmet.com'")
# fetch data from db one=one line, many=(int)lines, all=al lines
# print(c.fetchone())
# fetchmany(2)
# fetchall()

# save changes to db
conn.commit()
# close db
conn.close()