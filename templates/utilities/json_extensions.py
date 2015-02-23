import json

def get_json(data):
	return json.dumps(data, indent=4, sort_keys=True)


def file_get_json(filename):
	with open(filename, "r") as f: 
		return json.loads(f.read())

def file_write_json(filename, data):
	with open(filname, encoding='utf-8', mode='w') as v: 
		v.write(json.dumps(data, indent=4))
