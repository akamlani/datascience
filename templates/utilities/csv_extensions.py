import csv


def parse_file(datafile):
	data = []
	with open(datafile, "rb") as f:
		header = f.readline().split(",")
		for line in f:
			fields = line.split(",")
			entry = {}

			for i, value in enumerate(fields):
				entry[header[i].strip()] = value.strip()

			data.append(entry)
	return data

def parse_csv(datafile):
	with open(datafile, "rb") as f:
		reader = csv.DictReader(f, delimiter = ',')
		#csv.DictReader(open("name.csv"))
		header = reader.fieldnames
		#header = r.next()
		data = [row for row in reader]
	return data

def write_csv(datafile, data, header):
	with open(datafile, "wb") as f:
		writer = csv.DictWriter(f, delimiter=",", fieldnames= header)
		writer.writeheader()
		for row in data:
			writer.writerow(row)

def parse_tsv(datafile):
	with open(datafile, "rb") as f:
		reader = csv.DictReader(f, delimiter = '\t')
		for line in reader:
			data.append(line)
		return data


def open_zip(datafile):
    with ZipFile('{0}.zip'.format(datafile), 'r') as file_zip: 
    	file_zip.extractall()




"""
if __name__ == '__main__':
	import os
	import pprint

	DATADIR = ""
	DATAFILE = "*.csv"

	datafile = os.path.join(DATADIR, DATAFILE)
	d = parse_csv(datafile)
	pprint.pprint(d)
"""

"""
http://docs.python.org/2/library/csv.html
rows = item, columns = fields, cells = values
"""