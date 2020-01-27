import csv, os

# get data from any file
def get_data(file):
    lst = []
    if os.path.isfile(file):
        with open(file,'r+') as f:
            reader = csv.DictReader(f)
            for i in reader:
                lst.append([i['first_name'],i['last_name'],i['email']])
    else:
        insert_data(file,'w',lst)
        return get_data(file)
    return lst

# insert data to any file
def insert_data(file,mode,data):
    with open(file,mode) as f:
        field_names = ['first_name','last_name','email']
        writer = csv.DictWriter(f,fieldnames=field_names)
        if mode == 'a':
            writer.writerows(data)
        elif mode == 'w':
            writer.writeheader()
            writer.writerows(data)

def main():
    print(" 1 : add new element\n 2 : update a element \n 3 : delete a element\n 4 : show file\n 5 : exit")
    number = int(input('pls input a option :'))
    if number == 1:
        name = input('pls input a first name :')
        l_name = input('pls input alast name :')
        email = input('pls input a email :')
        data = get_data('test3.csv')
        if email in [j for i in data for j in i]:
            print('user already exist, pls try again:')
        else:
            dic = [{'first_name':name,'last_name':l_name,'email':email}]
            insert_data('test3.csv','a',dic)
    elif number == 2:
        email = input('pls Enter the e-mail address of the registration you want to update :')
        data = get_data('test3.csv')
        # print(data)
        l1 = []
        for i in data:
            if email in i[2]:
                name = input('pls input a new first name :')
                l_name = input('pls input a new last name :')
                email = input('pls input a new email :')
                l1.append({'first_name':name,'last_name':l_name,'email':email})
            else:
                l1.append({'first_name':i[0],'last_name':i[1],'email':i[2]})
        # print(l1)
        insert_data('test3.csv','w',l1)
    elif number == 3:
        email = input('pls Enter the e-mail address of the registration you want to delete :')
        data = get_data('test3.csv')
        # print(data)
        l1 = []
        for i in data:
            if email not in i[2]:
                l1.append({'first_name':i[0],'last_name':i[1],'email':i[2]})
            else:
                pass
        print(l1)
        insert_data('test3.csv','w',l1)
    elif number == 4:
        data = get_data('test3.csv')
        for i in data:
            print(i[0]+' '+i[1]+' '+i[2])
    elif number == 5:
        exit()
    else:
        print('invalid option')
    return main()
main()