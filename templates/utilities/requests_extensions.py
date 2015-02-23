import requests

def query(url, params, uid="", fmt="json"):
	r = requests.get(url + uid, params=params)
	print "requesting", r.url
	if(r.status_code == requests.codes.ok): 
		return r.json
	else: 
		r.raise_for_status()

	#r.content
	#user = json.loads(r.content)
	#print user.keys()

def post(url, data_params)
	r = requests.post(url, data=data_params)



