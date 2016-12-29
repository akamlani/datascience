### Dependency Libraries:
- Implementation based on Python 2.7.x
- can be installed via pip or conda (for ananconda) as appropriate:
    - e.g. pip install pandas
    - e.g. conda install pandas

Libraries:
- pandas, numpy, json
- argparse, os, datetime, re
- matplotlib, seaborn
- warnings, sys

### Usage:
first unzip directory into proper relative directory location
directory structure expects PASS, FAIL in {fixture_name}/data root directories

to use default data directory: fixture/data
python factory_metrics.py
to provide a different fixture directory:
python factory_metrics.py -f fixture_1_0
python factory_metrics.py -f /Users/jg/fixture_1_0
help usage options:
python factory_metrics.py -h

### Config
A configuration file is used to set the patterns and test to be performed.
Currently only the voltage test is being performed from the config file, but others can be added.

If no configuration file is found, default voltage patterns are used, noted by naming of output log file.
The default patterns are the same as in the config file.

- Dependency: {fixture_name}/config/config.json
- name: name of test (also name of file (voltage) to be created e.g. voltage_stats.txt)
- pattern: any string patterns to match against 'TEST' as a regular expression
- prefix:  any character to startwith to match against 'TEST' as a regular expression

### Images
An Images directory {fixture_name}/images/{date} will be created with the following types of files:
- units_failure_metrics.png: counts of all failure types recorded
- units_status_metrics.png: pass/failure metrics about a unit and/or experiment
- units_tested_datehour.png: units tested by hour for each day  
- units_tested_hourly.png: units tested by hour

### Logs
A Log directory {fixture_name}/logs/{date} will be created with the following files:
- {fixture_name}_rootdir_attr.txt: attributes of the root directory
- {fixture_name}_status_metrics.txt:  metrics per encoded files (reference definition of metrics)
- {fixture_name}_testfailuretype_stats.txt: counts of failure codes occured over ALL failure tests  
- {fixture_name}_units_served_hourly_summary.txt: descriptive hourly statistics per units served over all days
- {fixture_name}_units_served_hourly_stats.txt: breakdown of units served by hour
- {fixture_name}_units_served_datehourly.txt: breakdown of units served by hour per a particular day
- {fixture_name}_voltage_stats.txt: voltage rail breakdown statistics over all tests performed (Pass and Fail)

### Background on Metrics
The following Metrics will be printed out on execution:
- Root Dir Structure
- Units Served Hourly: Mean/Median/STD
- Top 10 Failure Types
- Unit Status/Failure Metrics

### Definition of Metrics
- num_units: number of unique units tested based on filename
Example:
filename  = LB1537330100624_WIEOTR_20161222_162445.csv<br>
unit_name = "LB1537330100624"<br>
timestamp = 20161222_162445 (Year:Month:Day:Hour:Minute:Second)
- num_tests: number of overall tests executed over all csv files
all csv files in PASS and FAIL directory paths
- num_pass_units, num_pass_units_pct, num_pass_tests, num_pass_tests_pct
- num_fail_units, num_fail_units_pct, num_fail_tests, num_fail_tests_pct
- num_units_overlapped_passfail:
if there is any corresponding overlap between sucess and failure units
only reported if the latest timestamp for the unit does not conform to success
- num_tests_multiple_failures
the number of tests that were executed multiple times for a unit
this includes all of the units that were tested multiple times due to failures
- num_units_multiple_failures
the number of unique units that had multiple failures
