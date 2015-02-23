from IPython.core.debugger import Pdb

def set_trace():
	'''place set trace directly in code as trace point'''
	# press c(continue) to have code resume
	Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def debug(f, *args, **kwargs):
	'''invoke debugger passing function and arguements (debug(f, args))'''
	from IPython.core.debugger import Pdb
	pdb = Pdb(color_scheme='Linux')
	return pdb.runcall(f, *args, **kwargs)

class Message:
	'''string representation of an object'''
	def __init__(self, arg):
		self.arg = arg
	def __repr__(self):
		return 'Obj: %s' % self.arg 




# ipython timing a function
# %time time_elapsed = <function evaluation>
# %timeit <function evaluation>

# ipython profiling 
# python -m cprofile -s cumulative <script.py>
# %run -p -s cumulative <script.py>
# %prun -s cumulative <function> ****

# ipython line profiling (line_profiler library via PyPI)
# via configuration: c.TerminalIPythonApp.extensions = ['line_profiler']
# %run prof_mod; %prun <function>
# %lprun -f <func> <statement to profile> ****


