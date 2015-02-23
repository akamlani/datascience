def iter_lines(input_data):
	for row in input_data: 
		for field, val in row.iteritems():
			 if field not in fields or empty_val(val):
	         	continue
	         	
def skip_lines(input_data, skip):
    for i in range(0, skip):
        next(input_data)

def empty_val(val):
    val = val.strip()
    return (val == "NULL") or (val == "")










