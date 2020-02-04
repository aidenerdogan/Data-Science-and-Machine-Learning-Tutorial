import csv,re

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


# # Find classification for all two class csv files
def find_class(class_index):
    class1 = None
    class2 = None
    class1 = get_data(file)[0][class_index]
    for row in get_data(file):
        if class1 == None and class2 == None:
            break
        elif row[class_index] != class1:
            class2 = row[class_index]
    # print(class1,class2)
    return class1,class2


# basic  nlp func1
def algorithm1():
    class1 = []
    class2 = []
    notr = []
    f0,f1 = find_class(class_index)
    # f1 = find_class(class_index)[1]
    for row in get_data(file):
        # remove newline character and leading spaces
        l1 = row[text_index].strip()
        # convert to lowercase to avoid mismatch
        l1 = l1.lower()
        # split comment into the words(list) by everything(' ',',','.'...)
        words = re.findall(r"[\w']+",l1)
        if row[class_index] == f0:
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
    # for i in class1:
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
        # count class1(class2) keys
        if word in class1:
            c1 = c1 + 1
        # count not class1 (class1) keys
        elif word in class2:
            c2 = c2 + 1
    # if class1 keys of count more than not class1 keys
    if c1 > c2:
        print('this text ',round(c1/len(words)*100,2), 'percent ',f0)
    # if not class1 keys of count more than class1 keys
    elif c2 > c1:
        print('this text ',round(c2/len(words)*100,2), 'percent ',f1)
    else:
        print('this text is notr')
    # print(class1)
    # print(class2)
    # print(notr)
# create_nlp()

# basic nlp func2
def algorithm2():
    class1 = []
    class2 = []
    f0 = find_class(class_index)[0]
    f1 = find_class(class_index)[1]
    for row in get_data(file):
        l1 = row[text_index].strip()
        l1.lower()
        words = re.findall(r"[\w']+",l1)
        if row[class_index] == f0:
            for word in words:
                if word not in class1:
                    class1.append(word)
        else:
            for word in words:
                if word not in class2:
                    class2.append(word)
        # for word in words:
        #     if row[0] == 'class1':
        #         if word not in class2:
        #             class2.append(word)
        #     else:
        #         if word not in class1:
        #             class1.append(word)
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
        if word in class1:
            c1 = c1 + 1
        elif word in class2:
            c2 = c2 + 1
    # if class1 keys of count more than not class1 keys
    if c1 > c2:
        print('this text ', round(c1/len(words)*100,2),'percent ',f0)
    # if not class1 keys of count more than class1 keys
    elif c2>c1:
        print('this text ', round(c2/len(words)*100,2), 'percent ',f1)
    else:
        print('text is notr')

# main
if __name__=='__main__':
    file = input('pls input a csv file :')
    class_index = int(input('Please enter the index of the classifier term. :'))
    text_index = int(input('Please enter the index of the text term. :'))
    algorithm1()
    algorithm2()
    # print(find_class(class_index)[0])
