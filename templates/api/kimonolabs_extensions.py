import json
import urllib


def kimono_url(url, api_key=api_key):
	data=json.load(urllib.urlopen(url))

	# iterate through json object to collect data
	for n in xrange(data['count']):
		data['results']['collection1'][n]['title']['text']
		data['results']['collection1'][n]['rating']
		data['results']['collection1'][n]['sales']

	# make a dataframe out of the python dictionary values
	data = pd.DataFrame({})
	return data;

def test_vector(url):
	import yaml
	credentials = yaml.load(open('/home/akamlani/projects/datascience/classes/ga/ds/api_cred.yml'))
	api_key = credentials['kimono_key']

	url = "https://www.kimonolabs.com/api/{api_tab}?".format(api_tab=api) + \
		  "api_key={}.format(api_key)" 		 				  			  + \
		  "&release_date={year}".format(year=year)

	data = kimono_url(url, api_key)


"""
Examples:

//retrieve via json
GET api/athletes/{ATHLETE_ID}
results = json.load(urllib.urlopen("http://sochi.kimonolabs.com/api/athletes/{ATHLETE_ID}?apikey=<key>"))

//curl 
curl --include --request GET "https://www.kimonolabs.com/api/{YOUR_API_ID}?apikey={YOUR_API_KEY}"
"""

