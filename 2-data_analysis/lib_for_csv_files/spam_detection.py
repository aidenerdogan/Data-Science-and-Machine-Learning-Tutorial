import csv,re

# step-1 join datas
# don't use step-1 becuse i used for crate new_spam.csv file.

# file_list = ['Youtube01-Psy.csv','Youtube02-KatyPerry.csv','Youtube03-LMFAO.csv','Youtube04-Eminem.csv','Youtube05-Shakira.csv']
# def insert_data(file):
#   with open('spam.csv','a',encoding="ISO-8859-1",newline='') as csv_file:
#       csv_writer = csv.writer(csv_file,delimiter=',')
#       for i in file:
#           csv_writer.writerow(i)
#       # csv_writer.writerows(file)
# def join():
#   for i in file_list:
#   insert_data(get_data(i))
# def get_data(file):
#   lst = []
#   with open(file,'r',encoding="ISO-8859-1",newline='') as csv_file:
#       csv_reader = csv.reader(csv_file,delimiter=',')
#       for row in csv_reader:
#           if row[4] == 1.0:
#               row[4] = 'spam'
#           else:
#               row[4] = 'ham'
#           tmp =[row[4],row[3],'','','']
#           lst.append(tmp)
#   return lst


# step-2 data analysis
def get_data(file):
    # open csv file with ISO encoding(secially fot this data)
    with open(file,'r',encoding="ISO-8859-1") as csv_file:
        # create a reader object
        csv_reader = csv.reader(csv_file,delimiter=',')
        # for row in csv_reader.head():
        #     print(row)
        # return like a list 
        return list(csv_reader)[1:]

# for i in get_data():
#     print(type(i[0]))

# nlp script 

# basic  nlp func1
def algorithm1():
    spam = []
    not_spam = []
    notr = []
    for row in get_data(file):
        # remove newline character and leading spaces
        l1 = row[1].strip()
        # convert to lowercase to avoid mismatch
        l1 = l1.lower()
        # split comment into the words(list) by everything(' ',',','.'...)
        words = re.findall(r"[\w']+",l1)
        if row[0] == 'spam':
            for word in  words:
                # create unique spam class
                if word not in spam and word not in not_spam:
                    spam.append(word)
                # create unique notr class
                elif word not in notr and word not in spam:
                    notr.append(word)
        else:
            for word in words:
                # cleare unique not_spam class
                if word not in spam and word not in not_spam:
                    not_spam.append(word)
                # continue to create notr class
                elif word not in notr and word not in not_spam:
                    notr.append(word)
    # for i in spam:
    #     print(i)
    text = input('1pls input a text :')
    # remove newline character and leading spaces
    l1 = text.strip()
    # convert to lowercase to avoid mismatch
    l1 = l1.lower()
    # split comment into the words(list) by everything(' ',',','.'...)
    words = re.findall(r"[\w']+",l1)
    c1 = 0
    c2 = 0
    for word in words:
        # count spam(negative) keys
        if word in spam:
            c1 = c1 + 1
        # count not spam (positive) keys
        elif word in not_spam:
            c2 = c2 + 1
    # if spam keys of count more than not spam keys
    if c1 > c2:
        print('this text ',round(c1/len(words)*100,2), 'percent spam')
    # if not spam keys of count more than spam keys
    elif c2 > c1:
        print('this text ',round(c2/len(words)*100,2), 'percent not spam')
    else:
        print('this text is notr')
    # print(spam)
    # print(not_spam)
    # print(notr)
# create_nlp()

# basic nlp func2
def algorithm2():
    positive = []
    negative = []
    for row in get_data(file):
        l1 = row[1].strip()
        l1.lower()
        words = re.findall(r"[\w']+",l1)
        if row[0] == 'spam':
            for word in words:
                if word not in negative:
                    negative.append(word)
        else:
            for word in words:
                if word not in positive:
                    positive.append(word)
        # for word in words:
        #     if row[0] == 'spam':
        #         if word not in negative:
        #             negative.append(word)
        #     else:
        #         if word not in positive:
        #             positive.append(word)
    text = input('2pls input a text :')
    # remove newline character and leading spaces
    l1 = text.strip()
    # convert to lowercase to avoid mismatch
    l1 = l1.lower()
    # split comment into the words(list)
    words = re.findall(r"[\w']+",l1)
    c1 = 0
    c2 = 0
    for word in words:
        if word in positive:
            c1 = c1 + 1
        elif word in negative:
            c2 = c2 + 1
    # if spam keys of count more than not spam keys
    if c1 > c2:
        print('this text ', round(c1/len(words)*100,2),'percent not spam')
    # if not spam keys of count more than spam keys
    elif c2>c1:
        print('this text ', round(c2/len(words)*100,2), 'percent spam')
    else:
        print('text is notr')

# nlp_option2()
if __name__=='__main__':
    file = input('pls input a csv file :')
    algorithm1()
    algorithm2()