import xlrd
import xlwt

def parse_excel_file(datafile, idx=0):
	workbook = xlrd.open_workbook(datafile)
	sheet = workbook.sheet_by_index(idx)
	
	data = [[sheet.cell_value(r, col) 
            for col in range(sheet.ncols)] 
                for r in range(sheet.nrows)]

    """
 	for row in range(sheet.nrows):
        for col in range(sheet.ncols):
                print sheet.cell_value(row, col)
    """
    return data


def log_excel_file(datafile, idx=0):
	workbook = xlrd.open_workbook(datafile)
	sheet = workbook.sheet_by_index(idx)

	print sheet.nrows
	print sheet.cell_type(row, col)
	print sheet.cell_value(row, col)

	cv = sheet.col_values(col, start_rowx=row, end_rowx=None)
	maxval = max(cv)
	maxpos  = cv.index(maxval) + 1
	maxcell = sheet.cell_value(maxpos, 0)
	minval = min(cv)


def default_excel_file(datafile, sheet_name):
	wb = Workbook()
	ws0 = wb.add_sheet(sheet_name)
	wb.save(datafile)


"""
if __name__ == '__main__':
	import os
	import pprint

	DATADIR = ""
	DATAFILE = "*.xlsx"

	datafile = os.path.join(DATADIR, DATAFILE)
	d = parse_excel_file(datafile)
	print log_excel_file(datafile)
	pprint.pprint(d)
"""
   

