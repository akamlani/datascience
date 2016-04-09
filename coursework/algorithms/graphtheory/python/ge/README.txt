[GE]
Author: Ari Kamlani
Release Date: Dec 05 2015
Version 1.0

Goal: Spectral Analysis on GE dataset to find structure and grouping in the underlying de-identified dataset.

File Input: ge_data.csv
File Output:  ouput/*
File Output:  ge_spectralcluster_analysis.xlsx
File Capture: graphs_ge_project.html (this is an html capture of the ipython notebook file)

Output Analysis Files given:
ge_spectralcluster_analysis_defaultcase.xlsx        [Other than X9 (Categorical), all features included]
ge_spectralcluster_analysis_featureselection.xlsx   [Produced for removing Features X15,X16,X17,X18,X19]
output_default/*.text                               [Other than X9 (Categorical), all features included]
output_featureselect/*.txt                          [Produced for removing Features X15,X16,X17,X18,X19]

output*/* files are the cluster produced for a given k value (2..10).  A separate file for ech value of k
is created.  The output here is what indices are assigned to a particular cluster, for a given value of k.
Indicies are zero based.

ge_spectral_analysis_*.xlsx are produced directly from the python code, rather than manually copying
the entries into it.  this way excel files can be reproduced by any changes in the algorithm or dataset.
There are only a few minor manual adjustments made by hand afterwards to beautify it.  A separate tab
is created for different values of k.  Within each tab, clusters are identified by color coding and a
label is noted on most left-sided column.

To Execute Code:
graphs_ge_project.ipynb  (originally coded file using ipython notebook, reference [iPython Notebook section])
graphs_ge_project.py     (python file implementation download) - self contained other than input csv file.


[General Python dependencies]
Python Dependency Libraries:
os:     to determine if a given directory exists or needs to be created on the filesystem
pandas: for dataframe manipulation, and logging to *.xlsx files where appropriate
numpy:  for underlying high performance matrix manipulation of dataframes
numpy/scipy: for eigenvector, eigenvalue calculation
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
