[Metro Map]
Author: Ari Kamlani
Release Date: Nov 30 2015
Version: 1.0

Goal:
Dijkstra Algorithm to find the shortest path via implemented Priority Queue option from a starting
metro station to metro station destination.  These are controlled by the input files noted below.

File Input:  metro_input.txt, metro_complet.txt
File Output: metro_output.txt
File Capture: graphs_metro.html (this is an html capture of the ipython notebook file for viewing purposes)

Modify the metro_input.txt for configuration of the program.
View the output in ipython notebook(jupyter) or the output from metro_output.txt

File Descriptions:
metro_input.txt:
Sections to be modified (note, any items marked with '#' in front of it will be identified as a comment):
[source station]:      Origininating Station
[destination station]: Destination Station
[Priority Queue Type]: Priority Queue Types = {Fibonacci, unordered array}

metro_output.txt:
Header: ['station', 'station name', 'predecessor', 'predecessor name', 'cost']
Followed by station detail from source to destination, listing the cost at each stop

To Execute Code:
graphs_metro.ipynb  (originally coded file using ipython notebook, reference [iPython Notebook section])
graphs_metro.py     (python file implementation download)


[General Python dependencies]
Python Dependency Libraries:
pandas: for dataframe manipulation, and logging to *.xlsx files where appropriate
numpy:  for underlying high performance matrix manipulation of dataframes
pprint: for pretty print logging style
codecs: for ISO format of encoding/decoding strings
sys:    for the large numbers identification in python (representing value for ~infinity)
fibonacci-heap-mod: for implementation of fibonacci heap used in Dijkstra algorithm
numpy/scipy: for eigenvector, pairwise distance calculations

Python Interpreter (version 2.7):
Anaconda Distribution can be installed via the following instructions:
http://docs.continuum.io/anaconda/install

Anaconda should come with the pandas and numpy libraries included by default.
To install additional modules, use: conda install <package name>
If the package is not found, it can be installed via pip in python: pip install <package name>

[iPython Notebook]
iPython notebook:
Place the data files and the *.ipynb files in the same directory
Launch ipython notebook (via command 'ipython notebook') command in the terminal, click on the appropriate *.ipynb file
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