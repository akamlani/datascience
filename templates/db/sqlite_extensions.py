import sqlite3


def create(filename, attributes):
	# Create empty database
	open(filename, attributes).close()
	# Create empty sql table in database
	db = sqlite3.connect(filename)
	conn = db.cursor()
	return conn

def create_table(tablename, parameters):
	conn.execute("""CREATE TABLE logins(
				    timestamp text,
				    date text,
				    hour int
				    )""")


def insert_row(tablename, arguements):
    conn.execute('INSERT INTO logins VALUES (?,?,?)', (ts,temp_date,temp_hour))
	db.commit()


def query_fetch_results(tablename):
	conn.execute("SELECT date, hour, COUNT(*) FROM logins GROUP BY date, hour ORDER BY COUNT(*) DESC")
	results = conn.fetchall()
	return results

def insert_batch(tablename, df):
    logins_tuple = zip(iter(logins))
    db.executemany('INSERT INTO Logins(TimeStamp) VALUES (?)', logins_tuple)
    conn.commit()



"""
filename = './logins.db'
attributes = 'w'
tablename = ""CREATE TABLE logins(timestamp text, date text, hour int)""
insert_query = 'INSERT INTO logins VALUES (?,?,?)', (ts,temp_date,temp_hour)
fetch_query = "SELECT date, hour, COUNT(*) FROM logins GROUP BY date, hour ORDER BY COUNT(*) DESC"
"""

""" 
commands:
.isin([])										# match with list
.max(), .min()									# select the smallest/largest from dataframe
.startswith(), .endswith(), .contains()			# string match
"""
