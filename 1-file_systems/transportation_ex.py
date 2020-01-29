import csv

def get_data(file):
	l1 = []
	with open(file,'r',newline='') as cvs_file:
		csv_reader = csv.reader(cvs_file,delimiter=',')
		for i in csv_reader:
			l1.append(i)
			# print(i)
	return l1
def insert_data(file,mode,data):
    with open(file,mode,newline='') as f:
        # creating a csv dict writer object
        writer = csv.writer(f,delimiter=',')
        writer.writerows(data)

reading_f1 = 'SalesJan2009.csv'
reading_f2= 'TechCrunchcontinentalUSA.csv'
writing_f1 = 'test4.csv'
writing_f2 = 'test5.csv'

# get_data('TechCrunchcontinentalUSA.csv')
insert_data(writing_f1,'w',get_data(reading_f1))
insert_data(writing_f2,'w',get_data(reading_f2))

# print(get_data('test1.csv'))