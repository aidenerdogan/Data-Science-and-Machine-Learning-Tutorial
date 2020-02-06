import csv,re
def get_data(f):
	# open csv file like a obj.
	with open(f,'r',encoding="ISO-8859-1") as file:
		csv_reader = csv.DictReader(file,delimiter=',')
		# print(csv_reader.fieldnames)
		for row in csv_reader:
			yield row
def get_count():
	types = dict()
	listed = dict()
	countries = dict()
	for row in get_data('netflix_titles.csv'):
		if row['type'] in types:
			types[row['type']] = types[row['type']] + 1
		else:
			types[row['type']] = 1
		words1 = row['listed_in'].split(",")
		for k in words1:
			k = k.strip()
			if k in listed:
				listed[k] = listed[k] + 1
			else:
				listed[k] = 1
		# l1 = row['country'].strip()
		# l1 = l1.lower()
		words2 = row['country'].split(',')
		for k in words2:
			k = k.strip()
			if k in countries:
				countries[k] = countries[k] + 1
			else:
				countries[k] = 1
	print(sorted(types.items(), key = 
	             lambda kv:(kv[1], kv[0])))
	print(sorted(listed.items(), key = 
	             lambda kv:kv[1]))
	print(sorted(countries.items(), key = 
	             lambda kv:kv[1]))
def find_film(text):
	film_list = []
	# print(re.findall(r"[\w']+",text))
	for e,row in enumerate(get_data('netflix_titles.csv')):
		lst = []
		del row['show_id']
		del row['date_added']
		del row['rating']
		del row['duration']
		# print(row.values())
		for i in row.values():
			# print(i.split())
			# lst.append(i.split())
			[lst.append(k) for k in i.split()]
			# lst.append(i)
		c = 0.0
		for k in re.findall(r"[\w']+",text):
			if k in lst:
				c = c + 1
		# print(c)
		if c > len(re.findall(r"[\w']+",text))/3:
			film_list.append(row['title'])
	print(film_list)

text = input('Please enter a phrase about your movie tastes :')
find_film(text)
