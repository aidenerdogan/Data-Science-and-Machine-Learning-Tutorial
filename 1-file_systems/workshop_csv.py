import csv

def insert_data(file_name,mode,l1):
    with open(file_name,mode) as f:
        for i in l1:
            f.write(i)