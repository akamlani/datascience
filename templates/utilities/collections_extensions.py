from collections import defaultdict


def create_sorted_dict():
	d = defaultdict(int)
	keys = d.keys()
	keys = sorted(keys, key=lambda s: s.lower())
	for k in keys: v = d[k]; print "%s: %d" % (k,v)