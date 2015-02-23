%matplotlib inline
import matplotlib.pyplot as plt  

def line_graph(x, y):
	fig=figure() 	
	plot(x,y)		

def scatter_plot(x,y):
	fig=figure()	
	scatter(x,y)			

def configure_plot(plt, title_name, filename, x_label, y_label, legend):
	plt.title(title_name)
	xlabel(x_label)
	ylabel(y_label)
	plt.legend(loc='best')
	plt.grid()
	return plt
	



# isinteractive(): check interactive mode
# ion(), ioff(): turn interactive mode on and off
# draw(): forces a figure redraw

# http://matplotlib.org/api/axis_api.html#matplotlib.axis.Tick
