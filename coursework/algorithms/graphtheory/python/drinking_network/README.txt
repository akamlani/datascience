
[Drinking Network]
Author: Ari Kamlani
Release Date: Nov 30 2015
Version 1.0

Goal: BFS to determine if a queried pipe label is associated with which valves, in case their is a problem
with a given pipe, e.g. leak, break, etc.

File Input: drinking_network.inp, drinking_network_input.txt
File Output: drinking_network_output.txt
File Capture: graphs_drinking_network.html (this is an html capture of the ipython notebook file for viewing purposes)

Modify the drinking_network_input.txt for configuration of the program.
View the output in ipython notebook(jupyter) or the output from drinking_network_output.txt

File Descriptions:
drinking_network_input.txt:
Sections to be modified (note, any items marked with '#' in front of it will be identified as a comment):
[Pipe Queries]:      Pipes to be queries (each to be separated on a new line)

drinking_network_output.txt:
Header: ['pipe', 'valves']
Followed by pipe/valves association detail, multiple valves may be given in some cases

To Execute Code:
graphs_drinking_network.ipynb  (originally coded file using ipython notebook, reference [iPython Notebook section])
graphs_drinking_network.py     (python file implementation download)


[General Python dependencies]
Python Dependency Libraries:
pandas: for dataframe manipulation, and logging to *.xlsx files where appropriate
numpy:  for underlying high performance matrix manipulation of dataframes
pprint: for pretty print logging style

Python Interpreter (version 2.7):
Anaconda Distribution can be installed via the following instructions:
http://docs.continuum.io/anaconda/install

Anaconda should come with the pandas and numpy libraries included by default.
To install additional modules, use: conda install <package name>
If the package is not found, it can be installed via pip in python: pip install <package name>

[iPython Notebook]
iPython notebook:
Place the data files and the *.ipynb files in the same directory
Launch ipython notebook (via ipython notebook) command in the terminal, click on the appropriate *.ipynb file
From the Menu, click Cell->Run all

ipython interpreter:
Alternatively just execute the script once the text files have been modified: (ipython is preferred over python)
No arguements are required to execute, as all options are read from input files when appropriate.
python <script>


[Comments]
* Note the *.py files were downloaded from within the context of ipython notebook, so the cells In/Out will
be marked and commented out appropriately.  It is better to execute the *.ipynb file, so the html head dataframes
can be seen during each cell stage, if instructed to do so.  However in terms of final output, the same results
are produced by the *.py file.