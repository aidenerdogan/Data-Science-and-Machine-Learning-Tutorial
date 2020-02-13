# -*- coding: utf-8 -*-
"""pnds.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15VDLjiQpCTPw1cvP1mjXtdhNUdcCNkw7
"""

# import pandas
import pandas as pd

"""Pandas Series
 pandas.Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)

Parameters

    dataarray-like, Iterable, dict, or scalar value

        Contains data stored in Series.

        Changed in version 0.23.0: If data is a dict, argument order is maintained for Python 3.6 and later.
    indexarray-like or Index (1d)

        Values must be hashable and have the same length as data. Non-unique index values are allowed. Will default to RangeIndex (0, 1, 2, …, n) if not provided. If both a dict and index sequence are used, the index will override the keys found in the dict.
    dtypestr, numpy.dtype, or ExtensionDtype, optional

        Data type for the output Series. If not specified, this will be inferred from data. See the user guide for more usages.
    namestr, optional

        The name to give to the Series.
    copybool, default False

        Copy input data.
Source:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
"""

# create a serie
label = ['This','is','AI','path']
num = [1,2,3,4]
# they must be same size
pd_serie = pd.Series(label,num)
pd_serie
# check the data type

# and another way
pd_serie1 = pd.Series(label,[6,7,8,9])
pd_serie1

exam1 = {'ahmet':50,'mike':60,'rania':75}
S1 = pd.Series(exam1)
S1

exam2 = {'ahmet':90}
S2 =pd.Series(exam2)
S2

exam3 = {'chirag':100}
S3 = pd.Series(exam3)
S3

S1+S2 #-,/,* ...
# NaN = Not a numbe

exam4 = S3.append(S1)
exam4

S1['ahmet'],exam4['mike']

"""Pandas DataFrame
 
 class pandas.DataFrame(data=None, index: Optional[Collection] = None, columns: Optional[Collection] = None, dtype: Union[str, numpy.dtype, ExtensionDtype, None] = None, copy: bool = False)[source]

    Two-dimensional, size-mutable, potentially heterogeneous tabular data.

    Data structure also contains labeled axes (rows and columns). Arithmetic operations align on both row and column labels. Can be thought of as a dict-like container for Series objects. The primary pandas data structure.

    Parameters

        datandarray (structured or homogeneous), Iterable, dict, or DataFrame

            Dict can contain Series, arrays, constants, or list-like objects.

            Changed in version 0.23.0: If data is a dict, column order follows insertion-order for Python 3.6 and later.

            Changed in version 0.25.0: If data is a list of dicts, column order follows insertion-order for Python 3.6 and later.
        indexIndex or array-like

            Index to use for resulting frame. Will default to RangeIndex if no indexing information part of input data and no index provided.
        columnsIndex or array-like

            Column labels to use for resulting frame. Will default to RangeIndex (0, 1, 2, …, n) if no column labels are provided.
        dtypedtype, default None

            Data type to force. Only a single dtype is allowed. If None, infer.
        copybool, default False

            Copy data from inputs. Only affects DataFrame / 2d ndarray input.

source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
"""

# firstly crea a data
from numpy.random import randn
# create dataframe
df1 = pd.DataFrame(data = randn(5,5),index=['1','2','3','4','5'],columns=['co1','co2','co3','co4','co5'])
df1

# read any column/columns
df1[['co1','co2']]

# add any column/columns
df1['co6']= pd.Series(randn(5),['1','2','3','4','5'])
df1

# add any column/columns
df1['co7'] =(df1['co1']+ df1['co4']-df1['co3']*df1['co5'])/df1['co6']
df1

# exit/drop any column/columns
df1.drop('co3',axis=1,inplace=True)
df1
# axis=0:row,axis=1:column
# inplace= True:permanently delete,False:delete temporarily

# seta column name to index name
df1.set_index('co6',inplace=True)
df1

# df1.index.names
df1.columns.names

# 'loc' and 'iloc'
df1.loc['5'] #by column name
df1.iloc[4] #by column index (start from 0)

df1.loc['2','co7']
df1.loc[['2','co7'] and ['4','co6']] #or...
df1.loc[['2','co7'], ['col2','co6']]

df1
df1>1

boolDf = df1>0
boolDf

df1[boolDf]
# NaN mean False

df1[df1>0.5]

df1['co4']>0

df1[df1['co2']>0]

df1[df1['co4']>0]

df1[df1['co6']>0]

df1[df1['co1']>0]

df1[(df1['co2']>0)&(df1['co7']>0)]

df1[(df1['co2']>0)|(df1['co7']>1)]

# summary: Values that are 'true' are displayed and you can't use and/or instead of that |,&

df1

df1['co7'] = ['NewVal1','NewVal2','NewVal3','NewVal4','NewVal5',]

df1.set_index('co7',inplace=True)

df1.index.names

df1.columns.names

# Multi Index
OuterIndex = ['Group1','Group1','Group1','Group2','Group2','Group2','Group3','Group3','Group3']
InnerIndex = ['Index1','Index2','Index3','Index1','Index2','Index3','Index1','Index2','Index3']
list(zip(OuterIndex,InnerIndex))

hierarchy = list(zip(OuterIndex,InnerIndex))
hierarchy = pd.MultiIndex.from_tuples(hierarchy)
hierarchy

df2 = pd.DataFrame(randn(9,3),hierarchy,columns=['co1','co2','co3'])
df2

df2['co3']

# df2.loc['Group2']
df2.loc[['Group2','Group1']] # same df2.loc[['Group1','Group2']]

df2.loc['Group2'].loc['Index3']
df2.loc['Group2'].loc['Index3']['co2']

df2.iloc[2]

df2.index.names =['Groups','Indexes']
df2

df2.xs('Group2')
# xs same loc/iloc (both of)

df2.xs('Group1').xs('Index1')
df2.xs('Group1').xs('Index1').xs('co2')

df2.xs('Index1',level='Indexes')

# Lost or Miss Datas
# create a miss data
import numpy as np
arr = np.array([[10,20,np.nan],[3,np.nan,np.nan],[13,np.nan,4]])
# np.nan creating Not a Number value
arr

df = pd.DataFrame(arr,index=['Ind1','Ind2','Ind3'], columns=['co1','co2','co3'])
df

df.dropna()
# If there is at least one 'NaN' in the 'Index' line, it will be deleted.

df.dropna(axis=1)
# Not permanent unless you give 'place' value 'True'

df.dropna(thresh=2) #1,2,3 as you like you can give the num.
# 'Not Deleting if there are at least two smooth data'

df.fillna(value=0)#you can use string also

"""Let us assume that the structure of the data is appropriate and that the data we are looking for is independent of the mean, and we assign the mean to the 'NaN' values."""

df.sum()
df.sum().sum()

df.fillna(value=(df.sum().sum())/5)
df.fillna(value=(np.var(df))) #or np.std(df)

df.size

df.isnull()
# True mean NaN

df.isnull().sum() #for each column
df.isnull().sum().sum() #totalt count of NaN

df.size - df.isnull().sum() #count of each coulumn not NaN value

# GroupBy Operatisons

data = {'Job': ['Data Mining','CEO','Lawyer','Lawyer','Data Mining','CEO'],'Labouring': ['Immanuel','Jeff','Olivia','Maria','Walker','Obi-Wan'], 'Salary': [4500,30000,6000,5250,5000,35000]}

df = pd.DataFrame(data)
df

salaries = df.groupby('Salary')
salaries

salaries.sum()

# salaries.min()
salaries.max()

# summary
df.groupby('Salary').sum()

df.groupby('Job').sum().loc['CEO']

df.groupby('Job').count()

df.groupby('Job').min()
df.groupby('Job').min()['Salary']
df.groupby('Job').min()['Salary']['Lawyer']
df.groupby('Job').mean()['Salary']['CEO']

# Concatenate Merge Ve Join Funcs
# * Concatenate: join proccess
data = {'A': ['A1','A2','A3','A4'],'B': ['B1','B2','B3','B4'],'C': ['C1','C2','C3','C4']}
data1 = {'A': ['A5','A6','A7','A8'],'B': ['B5','B6','B7','B8'],'C': ['C5','C6','C7','C8']}
df1 = pd.DataFrame(data, index = [1,2,3,4])
df2 = pd.DataFrame(data1, index = [5,6,7,8])
df1

pd.concat([df1,df2])

pd.concat([df1,df2],axis=1)

# * Join
data = {'A': ['A1','A2','A3','A4'],'B': ['B1','B2','B3','B4'],'C': ['C1','C2','C3','C4']}
data1 = {'A': ['A5','A6','A7','A8'],'B': ['B5','B6','B7','B8'],'C': ['C5','C6','C7','C8']}
df1 = pd.DataFrame(data, index = [1,2,3,4])
df2 = pd.DataFrame(data1, index = [5,6,7,8])
df2

data2 = [[1,'K','L'],[2,'M','N'],[6,'O','P'],[7,'Q','R'],[8,'S','T']]
# data2 = np.array(data2) IF YOU WANT U CAN'T USE
# data2.shape

df2 = pd.DataFrame(data2, index = [0,1,2,3,4],columns = ['id', 'Feature1', 'Feature2'])
df2

"""Pandas Join
 DataFrame.join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False) → 'DataFrame'[source]

    Join columns of another DataFrame.

    Join columns with other DataFrame either on index or on a key column. Efficiently join multiple DataFrame objects by index at once by passing a list.

    Parameters

        otherDataFrame, Series, or list of DataFrame

            Index should be similar to one of the columns in this one. If a Series is passed, its name attribute must be set, and that will be used as the column name in the resulting joined DataFrame.
        onstr, list of str, or array-like, optional

            Column or index level name(s) in the caller to join on the index in other, otherwise joins index-on-index. If multiple values given, the other DataFrame must have a MultiIndex. Can pass an array as the join key if it is not already contained in the calling DataFrame. Like an Excel VLOOKUP operation.
        how{‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘left’

            How to handle the operation of the two objects.

                left: use calling frame’s index (or column if on is specified)

                right: use other’s index.

                outer: form union of calling frame’s index (or column if on is specified) with other’s index, and sort it. lexicographically.

                inner: form intersection of calling frame’s index (or column if on is specified) with other’s index, preserving the order of the calling’s one.

        lsuffixstr, default ‘’

            Suffix to use from left frame’s overlapping columns.
        rsuffixstr, default ‘’

            Suffix to use from right frame’s overlapping columns.
        sortbool, default False

            Order result DataFrame lexicographically by the join key. If False, the order of the join key depends on the join type (how keyword).

    Returns

        DataFrame

            A dataframe containing columns from both the caller and other.

source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html
"""

# Left Join
df1.join(df2)

df2.join(df1)

# Right Join
df1.join(df2,how='right')

df2.join(df1,how='right')

# Inner Join
df1.join(df2,how='inner') #same element different index for elements df2.join(df1,how='inner')

# Outer Join
df1.join(df2,how='outer') #same element different index for elements df2.join(df1,how='outer')

df1.join(df2,sort='True')

df1.join(df2,sort='False')

frames = [df1,df2]
df_keys =pd.concat(frames,keys=['x','y'])
df_keys

"""Pandas Merge
pandas.DataFrame.merge

DataFrame.merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None) → 'DataFrame'[source]

    Merge DataFrame or named Series objects with a database-style join.

    The join is done on columns or indexes. If joining columns on columns, the DataFrame indexes will be ignored. Otherwise if joining indexes on indexes or indexes on a column or columns, the index will be passed on.

    Parameters

        rightDataFrame or named Series

            Object to merge with.
        how{‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘inner’

            Type of merge to be performed.

                left: use only keys from left frame, similar to a SQL left outer join; preserve key order.

                right: use only keys from right frame, similar to a SQL right outer join; preserve key order.

                outer: use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically.

                inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys.

        onlabel or list

            Column or index level names to join on. These must be found in both DataFrames. If on is None and not merging on indexes then this defaults to the intersection of the columns in both DataFrames.
        left_onlabel or list, or array-like

            Column or index level names to join on in the left DataFrame. Can also be an array or list of arrays of the length of the left DataFrame. These arrays are treated as if they are columns.
        right_onlabel or list, or array-like

            Column or index level names to join on in the right DataFrame. Can also be an array or list of arrays of the length of the right DataFrame. These arrays are treated as if they are columns.
        left_indexbool, default False

            Use the index from the left DataFrame as the join key(s). If it is a MultiIndex, the number of keys in the other DataFrame (either the index or a number of columns) must match the number of levels.
        right_indexbool, default False

            Use the index from the right DataFrame as the join key. Same caveats as left_index.
        sortbool, default False

            Sort the join keys lexicographically in the result DataFrame. If False, the order of the join keys depends on the join type (how keyword).
        suffixestuple of (str, str), default (‘_x’, ‘_y’)

            Suffix to apply to overlapping column names in the left and right side, respectively. To raise an exception on overlapping columns use (False, False).
        copybool, default True

            If False, avoid copy if possible.
        indicatorbool or str, default False

            If True, adds a column to output DataFrame called “_merge” with information on the source of each row. If string, column with information on source of each row will be added to output DataFrame, and column will be named value of string. Information column is Categorical-type and takes on a value of “left_only” for observations whose merge key only appears in ‘left’ DataFrame, “right_only” for observations whose merge key only appears in ‘right’ DataFrame, and “both” if the observation’s merge key is found in both.
        validatestr, optional

            If specified, checks if merge is of specified type.

                “one_to_one” or “1:1”: check if merge keys are unique in both left and right datasets.

                “one_to_many” or “1:m”: check if merge keys are unique in left dataset.

                “many_to_one” or “m:1”: check if merge keys are unique in right dataset.

                “many_to_many” or “m:m”: allowed, but does not result in checks.

            New in version 0.21.0.

    Returns

        DataFrame

            A DataFrame of the two merged objects.

source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
"""

# to concat different size datasets
data1 = {'A':['A1','A2','A3'], 'B': ['B1','B2','B3'], 'Key': ['K1','K2','K3']}
data2 = {'X':['X1','X2','X3','X4'], 'Y': ['Y1','Y2','Y3','Y4'], 'Key': ['K1','K2','K3','K4']}
df1 = pd.DataFrame(data1,index = [1,2,3])
df2 = pd.DataFrame(data2,index = [1,2,3,4])
df1

df2

pd.merge(df1,df2,on='Key')

pd.merge(df1,df2,on='Key',how='right') #default is left and left,right,inner,outer...

pd.merge(df1,df2,on='Key',how='right', right_index=True) #default is left and left,right,inner,outer...

pd.merge(df1,df2,on='Key',how='left', left_index=True) #default is left and left,right,inner,outer...

pd.merge(df1,df2,on='Key',how='outer')

pd.merge(df1,df2, right_index=True, left_index=True, how='outer')

# DataFrame Operations And Pivot Table
data = {'Column1': [1,2,3,4,5,6], 'Column2': [1000,1000,2000,3000,3000,1000],'Column3': ['Mace Windu','Darth Vader','Palpatine','Kylo Ren','Rey','Obi-Wan']}
df = pd.DataFrame(data)
df

df.head()#first 5 elements but you can use head(num)

df.tail()#last 5 elements but you can use tail(num)

df.describe()

df.info()

df['Column2'].unique()

df['Column2'].nunique()

df['Column2'].value_counts()

df[df['Column1']>=2]

df[(df['Column1']>=2) & (df['Column2']==3000)]

def sqr(x):
  return x**2
df['Column2'].apply(sqr)

# or
df['Column2'].apply(lambda x:x**2)

# update
df['Column2'] = df['Column2'].apply(sqr)

df['Column3'].apply(len)

df.drop('Column3',axis=1)

df.index

df.index.names

data = {'Co1':[1,2,3,4,5,6],'Co2':[200,300,200,300,300,400],'Co3':['John Wick','Dart Vader','Joker','Thanos','Harly Quin','Andrew NG']}

df = pd.DataFrame(data,index=[0,1,2,3,4,5])

df.sort_values(by=['Co1','Co3'])

df.sort_values(by=['Co1','Co2'])

df.sort_values('Co2',ascending=True)#descending can use also

df.sort_values('Co2',ascending=False)

df.sort_values('Co2',kind='heapsort') #mergesort,quicksort...

df.sort_values('Co2',na_position='first') #last

df = pd.DataFrame({'Month': ['January','February','March','January','February','March','January','February','March'],'State':['New York','New York','New York','Texas','Texas','Texas','Washington','Washington','Washington'],
'moisture': [20,25,65,34,56,85,21,56,79]})
df

df.corr()

# Let's say we have 3 percent humidity in 3 different cities in 3 different months.
df.pivot_table(index='Month',columns='State',values='moisture')

df.pivot_table(index='State',columns='Month',values='moisture')

"""Pandas Read Data


dataset = pd.read_csv ('File_Path \ Data_Name.csv') # You can read this on your IDE on PC.
df = pd.read_csv ("../ input / Data_Name.csv") # Kaggle to write a kernel (but the update is sometimes not working)
dataset.to_csv ('Data_Name') # To convert the data back to csv
excelset = pd.read_excel ('Excels_Name.xlsx') # To read the Excel File
excelset.to_excel ('excelnewfile.xlsx') # Convert to Excel file
New = pd.read_html ('Datasetin urlsi.html') # To read data on the Internet
"""

# Research
# Pandas_Profiling()
# for libs: https://www.scipy.org/
