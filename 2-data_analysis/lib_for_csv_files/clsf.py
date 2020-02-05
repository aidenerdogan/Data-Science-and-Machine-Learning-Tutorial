""" READ ME!
This script created for spam detection from any two class csv files.
It is contains two different detection algorithms."""
import csv,re

def get_data(file):
    # open csv file with ISO encoding(secially fot this data)
    with open(file,'r',encoding="ISO-8859-1") as csv_file:
        # create a reader object
        csv_reader = csv.DictReader(csv_file,delimiter=',')
        # return like a list 
        return list(csv_reader)

# Find classification for all two class csv files
def find_class(class_column):
    class1 = class2 = None
    class1 = get_data(file)[0][class_column]
    for row in get_data(file):
        if class1 == None and class2 == None:
            break
        elif row[class_column] != class1:
            class2 = row[class_column]
    # print(class1,class2)
    return class1,class2

# basic  nlp func1
def algorithm1():
    class1,class2,notr = [],[],[]
    f1,f2 = find_class(class_column)
    for row in get_data(file):
        # remove newline character and leading spaces
        l1 = row[text_column].strip()
        # convert to lowercase to avoid mismatch
        l1 = l1.lower()
        # split comment into the words(list) by everything(' ',',','.'...)
        words = re.findall(r"[\w']+",l1)
        if row[class_column] == f1:
            for word in  words:
                # create unique class1 class
                if word not in class1 and word not in class2:
                    class1.append(word)
                # create unique notr class
                elif word not in notr and word not in class1:
                    notr.append(word)
        else:
            for word in words:
                # cleare unique class2 class
                if word not in class1 and word not in class2:
                    class2.append(word)
                # continue to create notr class
                elif word not in notr and word not in class2:
                    notr.append(word)
    text = input('1pls input a text :')
    # remove newline character and leading spaces
    l1 = text.strip()
    # convert to lowercase to avoid mismatch
    l1 = l1.lower()
    # split comment into the words(list) by everything(' ',',','.'...)
    words = re.findall(r"[\w']+",l1)
    c1 = c2 = 0
    # find the number of class1 and class2
    for word in words:
        # count class1(class2) keys
        if word in class1:
            c1 = c1 + 1
        # count not class1 (class1) keys
        elif word in class2:
            c2 = c2 + 1
    # whichever is greater, text is closer to that class
    # and calculate accuracy c1orc2/len(text)
    if c1 > c2:
        print('this text ',round(c1/len(words)*100,2), 'percent ',f1)
    elif c2 > c1:
        print('this text ',round(c2/len(words)*100,2), 'percent ',f2)
    else:
        print('this text is notr')

# basic nlp func2
def algorithm2():
    class1,class2 = [],[]
    f1,f2 = find_class(class_column)
    for row in get_data(file):
        l1 = row[text_column].strip()
        l1.lower()
        words = re.findall(r"[\w']+",l1)
        # create a class1 for f1 class type
        if row[class_column] == f1:
            for word in words:
                if word not in class1:
                    class1.append(word)
        # create a class for f2 cass type
        else:
            for word in words:
                if word not in class2:
                    class2.append(word)
    text = input('2pls input a text :')
    # remove newline character and leading spaces
    l1 = text.strip()
    # convert to lowercase to avoid mismatch
    l1 = l1.lower()
    # split comment into the words(list)
    words = re.findall(r"[\w']+",l1)
    c1 = c2 = 0
    for word in words:
        if word in class1:
            c1 = c1 + 1
        if word in class2:
            c2 = c2 + 1
    # if class1 keys of count more than not class1 keys
    if c1 > c2:
        print('this text ', round(c1/len(words)*100,2),'percent ',f1)
    # if not class1 keys of count more than class1 keys
    elif c2 > c1:
        print('this text ', round(c2/len(words)*100,2), 'percent ',f2)
    else:
        print('text is notr')

# main
if __name__=='__main__':
    file = input('pls input a csv file :')
    class_column = input('Please enter the column of the classifier term. :')
    text_column = input('Please enter the column of the text term. :')
    algorithm1()
    algorithm2()