#get data from any file
def get_data(file_name):
    with open(file_name,'r+') as f:
        file = f.readlines()
    return file
#insert data to any file
def insert_data(file_name,mode,l1):
    with open(file_name,mode) as f:
        for i in l1:
            f.write(i)
# main func
def main():
    print(" 1 : add new element\n 2 : update a element \n 3 : delete a element\n 4 : show file\n 5 : exit")
    number = int(input('pls input a option :'))
    if number == 1:
        file = get_data('info.txt')
        print(file)
        file2 = []
        for i in file:
            file2.append(i.split('\n')[0].split()[1])
        print(file2)
        name = str(input('pls input a name :'))
        email = str(input('pls input a email :'))
        if email in file2:
            print("Please enter another user already registered user")
        else:
            val = []
            val.append(name + ' ' + email+'\n')
            insert_data('info.txt','a+',val)
    elif number == 2:
        email = str(input('pls input a email :'))
        file1 = get_data('info.txt')
        file2 = []
        for i in file1:
            if email in i.split('\n')[0]:
                new_name = str(input('pls input a new name :'))
                new_email = str(input('pls input a new email :'))
                i = str(new_name +' '+ new_email+'\n')
                file2.append(i)
            else:
                file2.append(i)
        print(file2)
        insert_data('info.txt','w',file2)
    elif number == 3:
        email = str(input('pls enter the email address of the record you want to delete :'))
        file1 = get_data('info.txt')
        file2 = []
        for i in file1:
            if email not in i.split('\n')[0]:
                file2.append(i)
            else:
                pass
            insert_data('info.txt','w',file2)
    elif number == 4:
        file1 = get_data('info.txt')
        for i in file1:
            print(i.split('\n')[0])
    elif number == 5:
        exit()
    else:
        print('input valid input')
    return main()

if __name__=="__main__":
    
    main()