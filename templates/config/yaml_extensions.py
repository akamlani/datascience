import yaml

def load_config(file):
	'''file as *.yml extension'''
	credentials = yaml.load(open(file))
	return credentials