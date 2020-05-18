# import csv module for csv files
import csv

# get data from csv file
def getData():
  # open csv file
  with open('netflix_titles.csv','r') as csvFile:
    # create a csv reader object like a dictionary
    csvReader = csv.DictReader(csvFile)
    for row in csvReader:
      # we don't need these columns
      del row['show_id']
      del row['date_added']
      # define generators
      yield row

# get data by type (tv show,movie or both)
def getByType(typ):
  # this list will be only for selected type films
  lst = []
  for row in getData():
    row['type'] = row['type'].strip()
    if typ == 1:
      if 'TV Show' in row['type']:
        lst.append(row)
    elif typ == 2:
      if 'Movie' in row['type']:
        lst.append(row)
    elif typ == 3:
        lst.append(row)
  # sort descanding list by release_year
  lst.sort(reverse = True,key = lambda x: x['release_year'])
  return lst

# get data by film category afte get data by getByType() function
def getByListed(category):
  lst = []
  for i,row in enumerate(listed):
    # listed coming from main(global value)
    # select category from category list by number
    if i == category:
      category = row
  for row in getByType(typ):
    # split listed_in (cagories) because listed_in contains category more than one
    words1 = row['listed_in'].split(",")
    # get selected category
    if category in words1:
      lst.append(row)
  return lst

# get data by taing type afte get data by getByListed() func
def getByRating(rate):
  lst = []
  # use last data for classified data and classify by selected rating
  # these ratings for USA but you can use them for international
  # if you want you can change to number like family,adult...
  for row in getByListed(category):
    if rate == 1:
      if row['rating'] in ['G','TV-Y','TV-G']:
        lst.append(row)
    elif rate == 2:
      if row['rating'] in ['TV-Y7','TV-Y7-FV','TV-14']:
        lst.append(row)
    elif rate == 3:
      if row['rating'] in ['R','PG','PG-13']:
        lst.append(row)
    elif rate == 4:
      if row['rating']in ['TV-MA','TV-PG','NC-17']:
        lst.append(row)
    elif rate == 5:
      lst.append(row)
  return lst

# get films after classifications
def getFilms():
  print(rate)
  data = getByRating(rate)
  ln = len(data)
  # print(ln)
  # you can get more than 5 films but we need to show first films sorted by release year
  if ln > 5:
      for i,row in enumerate(data):
        if i < 6:
          print(row['title'])
  # if you have les than films model is completing from same type films selecting by release year.
  else:
    for i,row in enumerate(data):
        print(row['title'])
    for i,row in enumerate(getByType(typ)):
      if i <= (5-ln):
        print(row['title'])

if __name__=='__main__':
  typ = int(input(' 1- Tv Show \n 2- Movie \n 3- Both \nPlease input a type number :'))
  # this list for categories. contains all different categories by selected type
  listed = []
  for row in getByType(typ):
    words1 = row['listed_in'].split(",")
    for k in words1:
      k = k.strip()
      if k not in listed:
        listed.append(k)
  # show categs by selected type
  for i,row in enumerate(listed):
    print(i,row)
  category = int(input('Please input a category :'))
  rate = int(input(' 1- Child\n 2- Young\n 3- Family\n 4- Adult\n 5- Anything\n Please input a rate type :'))
  # and finaly get classified datas(films)
  getFilms()