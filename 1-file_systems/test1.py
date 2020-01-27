import csv

def get_data(file):
	csv_reader = None
	with open(file,'r') as cvs_file:
		csv_reader = csv.DictReader(cvs_file)
	return csv_reader
print(get_data('test1.csv'))
path = '/home/indianic/Documents/ahmet/github/DataScientistPath/1-file_systems/test1.csv'
for i in get_data('test1.csv').items():
	print(i)