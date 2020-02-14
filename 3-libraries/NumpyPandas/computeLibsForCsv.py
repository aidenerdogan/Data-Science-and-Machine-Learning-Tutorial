import profile
import csv
import pandas as pd
import numpy as np

def time_csv(file):
	data = csv.DictReader(open(file))
		# for row in data:
		# 	print(row)

def time_pandas(file):
	data = pd.read_csv(file)
	# print(data)

# # compute run time,calling funcs,for csv module
# profile.run("print(time_csv('random.csv')); print()")
# # compute run time,calling funcs, for pandas
# profile.run("print(time_pandas('random.csv')); print()")


"""Note: Csv.DictReader remains the quickest option generaly, but Not recommended for data analysis tasks. Possibly working with the JSON format is one such area where this method will be quite useful."""

"""
What are the differences between csv and pandas modules?
-Csv using dict key this is not good for data analysis or data analytics
but pandas conerting data from file to dataframe
-Csv only reading/writing csv files but for pandas csv,text,exel... are posible,
-You can use vectors,matrices,linear algebra operations like matrix
but for csv these are not posible.
-Pandas has integration with a lot of other libraries.

Which scenario selcet pandas?
-You can use the chunksize option, if you are dealing with a large dataset and you are low on RAM.
-For 2D data analysis
-When you need extracting selections form data
-When u need parsing data
-When u filtering by some conditions
-Summary when you're doing advanced data analysis

Which scenario select csv?
-perform mundane operations with data select csv
-For analysis not better bcs usind dict key but when u are working
on JSON data type it is better than pandas
"""

"------------------------------------------------------------------------------"

# CREATING CSV FILE with RANDOM DATA
def create_data(file_name):
	# setting number of rows for data
	N = 10000000
	# Creating DataFram(df) with A to H (8 columns) and between 999,999999 random integer type values
	df = pd.DataFrame(np.random.randint(999,999999,size=(N,7)),columns=list('ABCDEFG'))
	# Creating one column with float type values, using the uniform distribution
	df['H']=np.random.rand(N)
	# creating two additional columns with random strings
	df['I']=pd.util.testing.rands_array(10,N)
	df['J']=pd.util.testing.rands_array(10,N)
	# see data
	print(df)

	# export the dataframe to csv using comma delimiting
	df.to_csv(file_name,sep=',')
# create_data('random2.csv')

# COMPUTE RUN TIMES
import time
import dask.dataframe as ddf
# import datatable as dt

file = 'random2.csv'

# start_time = time.time()
# data = csv.DictReader(open(file))
# for row in data:
# 	print(row)
# print("csv.DictReader took %s seconds \n" % (time.time() - start_time))

# start_time = time.time()
# data = pd.read_csv(file)
# print(data)
# print("pd.read_csv took %s seconds \n" % (time.time() - start_time))

start_time = time.time()
data = pd.read_csv(file, chunksize=10000000)
print("pd.read_csv with chunksize took %s seconds \n" % (time.time() - start_time))

start_time = time.time()
data = ddf.read_csv(file)
print(data)
print("dask.dataframe.read_csv took %s seconds \n" % (time.time()-start_time))

# start_time = time.time()
# data = dt.fread(file)
# print("datatable.fread took %s seconds" % (time.time()-start_time))


"""
RANDOM.CSV: 93.5 KB (1000,10)
 csv.DictReader took 5.793571472167969e-05 seconds 
 
 pd.read_csv took 0.010068655014038086 seconds 
 
 pd.read_csv with chunksize took 0.0009567737579345703 seconds 
 
 dask.dataframe.read_csv took 0.02411341667175293 seconds

RANDOM1.CSV 486.5 MB (500000,10)
csv.DictReader took 4.863739013671875e-05 seconds 

pd.read_csv took 9.809984922409058 seconds 

pd.read_csv with chunksize took 0.16832447052001953 seconds 

dask.dataframe.read_csv took 0.02391648292541504 seconds 


RANDOM2.CSV 974.0 MB (1000000,10)
csv.DictReader took 6.031990051269531e-05 seconds 

pd.read_csv with chunksize took 0.002941131591796875 seconds 

dask.dataframe.read_csv took 0.031235218048095703 seconds
""" 

""" read csv file in python
import csv
pandas (0-1GB:changes according to memory)
pandas with chunksize parameter (more than 1 GB)
pandas astype parameter
dask.dataframe (best one for more than 1GB-100GB)
datatable  still have not looked into how well it is integrated with other libraries.
paratext fastest one but diffucult setup (not posible with pip or hombrew)"""
# https://medium.com/casual-inference/the-most-time-efficient-ways-to-import-csv-data-in-python-cc159b44063d
