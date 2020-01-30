import csv,sqlite3,os
from zipfile import ZipFile as zf

conn = sqlite3.connect('test4.db')

c = conn.cursor()

def insert_table(table,data):
    c.execute("""CREATE TABLE IF NOT EXISTS %s (COMMENT_ID text,
            AUTHOR text ,
            DATE real,
            CONTENT text,
            CLASS real)""" % (table))
    c.executemany("""INSERT INTO %s (COMMENT_ID,AUTHOR,DATE,CONTENT,CLASS) VALUES (?,?,?,?,?)""" % (table),(data))
    conn.commit()
# insert_table('test1')
# def insert_into(data)
# get data from any file
def get_data(file):
    lst = []
    with open(file,'r+') as f:
        # creating a csv dict reader object
        reader = csv.reader(f)
        for i in reader:
            # add to list (bcs you can't use outside dict reader object)
            lst.append(i)
    return lst[:2]

my_directory = os.getcwd()
# print(my_directory)
file_path = 'YouTube-Spam-Collection-v1.zip'

tables = []
files = []
def get_files(file_path):
    with zf(file_path,'r') as zip_files:
        zip_files.extractall()
        entries = os.listdir(my_directory+'/'+(file_path.split('.')[0]))
        for i in entries:
            tables.append(i.split('-')[0])
            files.append(my_directory+'/'+file_path.split('.')[0]+'/'+str(i))
    return tables,files
get_files(file_path)
for i in files:


# get_files(file_path)
# print(tables)
# print(files).

def insert_data():
    get_files(file_path)
    for i,j in zip(tables,files):
        insert_table(i,get_data(j))


def fetchall_data(tables):
    for i in tables:
        # c.execute('SELECT * FROM Youtube01Psy')
        # rows = c.fetchall()
        # for i in rows:
        #     print(i)
        c.execute("""SELECT * FROM %s""" % (i))
        rows = c.fetchall()
        for row in rows:
            print(row)
# insert_data()
# fetchall_data(tables)


# insert_table('Youtube01Psy',get_data('Youtube01-Psy.csv'))

c.close()
conn.close()