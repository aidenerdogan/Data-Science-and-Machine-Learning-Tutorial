import sqlite3
 
con = sqlite3.connect('test.db')
l1 = []
def fetc_spam(con):
 
	cursorObj = con.cursor()
 
	obj = cursorObj.execute('SELECT name from sqlite_master where type= "table"')
 
	for i in obj:
		cursorObj.execute("""SELECT * FROM %s""" % (i))
		rows = cursorObj.fetchall()
		for row in rows:
			if row[4] == 1.0:
				print(row[0],'Spam')
			else:
				print(row[0],'Not Spam')
 
fetc_spam(con)