import datetime as dt

def convert_to_time(logins):
	login_dates = []
	for ts in logins:
	    temp_date = dt.datetime.strptime(ts,'%Y-%m-%d %H:%M:%S')
	    login_dates.append(temp_date)

	return login_dates

