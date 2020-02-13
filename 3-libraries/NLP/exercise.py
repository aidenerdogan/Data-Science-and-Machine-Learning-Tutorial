import pandas as pd
import dask.dataframe as ddf
import numpy as np

# get data from current file
chunks = pd.read_csv('toxic_comment.csv',chunksize=50000)
lst = []
for i in chunks:
	lst.append(pd.DataFrame(i))
	i.head()
# print(lst)
# print(data.info(memory_usage='deep'))

"""print(data.info(memory_usage='deep'))

for dtype in ['float','int','object']:
    selected_dtype = data.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
import numpy as np
int_types = ["uint8", "int8", "int16"]
for it in int_types:
    print(np.iinfo(it))
# We're going to be calculating memory usage a lot,
# so we'll create a function to save us some time!
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)
# int converting
data_int = data.select_dtypes(include=['int'])
converted_int = data_int.apply(pd.to_numeric,downcast='unsigned')
print(mem_usage(data_int))
print(mem_usage(converted_int))
compare_ints = pd.concat([data_int.dtypes,converted_int.dtypes],axis=1)
compare_ints.columns = ['before','after']
compare_ints.apply(pd.Series.value_counts)

# float converting (i dont have float in my data)
# data_float = data.select_dtypes(include=['float'])
# converted_float = data_float.apply(pd.to_numeric,downcast='float')
# print(mem_usage(data_float))
# print(mem_usage(converted_float))
# compare_floats = pd.concat([data_float.dtypes,converted_float.dtypes],axis=1)
# compare_floats.columns = ['before','after']
# compare_floats.apply(pd.Series.value_counts)

optimized_data = data.copy()
optimized_data[converted_int.columns] = converted_int
# optimized_data[converted_float.columns] = converted_float
print(mem_usage(data))
print(mem_usage(optimized_data))

data_obj = data.select_dtypes(include=['object']).copy()
data_obj.describe()
# print(data_obj.describe())

# dow = data_obj.id
# print(dow.head())
# dow_cat = dow.astype('category')
# print(dow_cat.head())
# dow_cat.head().cat.codes
# print(dow_cat.head().cat.codes)

# print(mem_usage(dow))
# print(mem_usage(dow_cat))


converted_obj = pd.DataFrame()
for col in data_obj.columns:
    num_unique_values = len(data_obj[col].unique())
    num_total_values = len(data_obj[col])
    if num_unique_values / num_total_values < 0.5:
        converted_obj.loc[:,col] = data_obj[col].astype('category')
    else:
        converted_obj.loc[:,col] = data_obj[col]
print(mem_usage(data_obj))
print(mem_usage(converted_obj))
compare_obj = pd.concat([data_obj.dtypes,converted_obj.dtypes],axis=1)
compare_obj.columns = ['before','after']
compare_obj.apply(pd.Series.value_counts)"""



# change types 
# data[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]] = data[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].astype('int32')

# print(type(data['toxic']))
# data shape
# print(data.shape)
# read first 5 row by head()
# print(data.head())
# slice to sentences
# lstSlice1 = []
# for i in lst:
# 	slice1 = i['comment_text'].str.split()
# 	lstSlice1.append(slice1)
# print(print(lstSlice1))
# big data issue for my memory
 # i tried chunksize but didnt work
lstSlice2 = []
for i in lst:
	slice2 = i['comment_text'].str.split(expand=True)
	lstSlice2.append(slice2)
print(lstSlice2)
# from collections import Counter
# import matplotlib.pyplot as plt

# for words in lstSlice2:
# 	counts = dict(Counter(words).most_common(50000))
# 	labels, values = zip(*counts.items())
# 	# sort your values in descending order
# 	indSort = np.argsort(values)[::-1]
# 	# rearrange your data\
# 	labels = np.array(labels)[indSort]
# 	values = np.array(values)[indSort]
# 	indexes = np.arange(len(labels))
# 	bar_width = 0.35
# 	plt.bar(indexes, values)
# 	# add labels
# 	plt.xticks(indexes + bar_width, labels)
# 	plt.show()
from nltk.book import *
for words in lstSlice2:
	freqDist = FreqDist(words)
	words = freqDist.keys()
	freqDist.plot(10)

