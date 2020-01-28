import csv, os

# get data from any file
def get_data(file):
    lst = []
    if os.path.isfile(file):
        with open(file,'r+') as f:
            # creating a csv dict reader object
            reader = csv.reader(f)
            for i in reader:
                # add to list (bcs you can't use outside dict reader object)
                lst.append(i)
    else:
        insert_data(file,'w',lst)
        return get_data(file)
    return lst

# insert data to any file
def insert_data(file,mode,data):
    with open(file,mode) as f:
        # field names 
        field_names = ['first_name','last_name','email']
        # creating a csv dict writer object
        writer = csv.DictWriter(f,fieldnames=field_names)
        if mode == 'a':
             # writing data rows (only values not keys)
            writer.writerows(data)
        elif mode == 'w':
            # writing headers (field_names) 
            writer.writeheader()
            writer.writerows(data)