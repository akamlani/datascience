from __future__ import division
import pandas as pd
import numpy as np 
import argparse
import os 
import re
import json 
import sys 

import warnings
from datetime import datetime

import matplotlib.pyplot as plt 
import seaborn as sns 

warnings.filterwarnings("ignore")

### Create Tabular Structure
def get_folder_attrs(path):
    root_dir    = os.listdir(path)
    root_attr   = {name: os.path.isdir(path + name) for name in root_dir}
    root_dirs   = map(lambda (k,v): k,filter(lambda (k,v): v==1, root_attr.iteritems()) )
    root_files  = map(lambda (k,v): k,filter(lambda (k,v): v==0, root_attr.iteritems()) )
    n_rootdirs  = len(root_dirs)
    n_rootfiles = len(root_files)
    return {
        'root_dirs': root_dirs,
        'num_rootdirs':  n_rootdirs,
        'root_files': root_files,
        'num_rootfiles': n_rootfiles
    }

def extract_subfolder_data(subfolder, rootpath):
    root_data_dict = {}
    for (dirpath, dirnames, filenames) in os.walk(rootpath+subfolder):
        if len(filenames) > 1:
            k = dirpath.split(rootpath)[1].strip('data/')
            v = list( set([filename.split('.')[0] for filename in filenames]) )
            root_data_dict[k] = v
    return root_data_dict 


def create_units(status_dict, root_path):
    # create a series of records of units that were tested
    df_units = pd.DataFrame()
    for k,v in status_dict.iteritems():
        for item in v:
            unit_type, date_rec = [s.strip() for s in k.split("/")]
            file_name = "_".join(item.split("_")[:-2])
            ts_rec = "".join(item.split("_")[-2:])
            is_dir = os.path.isdir(root_path + k + item)
            if not is_dir: 
                ts_rec = datetime.strptime(ts_rec, '%Y%m%d%H%M%S')
                filename = root_path + k +'/' + item + '.csv'
                # create new format to tabulate structure    
                unit_name = file_name.split("_")[0].strip() if unit_type == 'FAIL' else file_name 
                unit_dict = {'file_name':file_name, 'unit_name': unit_name, 
                             'unit_status': unit_type, 'date_record': ts_rec}
                df_units = df_units.append(unit_dict, ignore_index=True)
    df_units['date'] = df_units.date_record.dt.date
    df_units['hour'] = df_units.date_record.dt.hour
    return df_units

def create_dir_structure():
    if not os.path.exists(data_path): 
        print("No Data Present")
        sys.exit()
    else: 
        if not os.path.exists(log_path): os.makedirs(log_path)    
        if not os.path.exists(image_path): os.makedirs(image_path)
        if not os.path.exists(config_path): print("\nNo Config File: Using default config\n")
        attrs = get_folder_attrs(fixture_path)
        params = {k:v for k,v in attrs.iteritems() if k != 'root_files'}
        filename = log_path + file_prefix + 'rootdir_attr.txt'
        pd.Series(params, name='attributes').to_csv(filename, sep='\t')
        print "Root Dir Attributes:"; print pd.Series(params, name='attributes')
    return attrs

### Aggregation Calculations 
def calc_agg_stats(df_units):
    df_unit_counts = df_units.groupby('unit_name')['unit_status'].count()
    df_mult_tests  = df_unit_counts[df_unit_counts > 1].sort_values(ascending=False)
    df_mult_failures = df_units[(df_units.unit_name.isin(df_mult_tests.index)) & (df_units.unit_status == 'FAIL')]
    # aggregate statistics
    n_units, n_tests  = len(df_units.unit_name.unique()), len(df_units.unit_name)
    n_units_mult_failures, n_mult_failures  = (len(df_mult_tests), df_mult_tests.sum())
    # executed tests that are passing and failing 
    n_pass_tests, n_fail_tests = df_units.unit_status.value_counts()
    n_pass_tests_pct, n_fail_tests_pct = n_pass_tests/n_tests, n_fail_tests/n_tests
    # there are some boards that show up both in pass and failure ('LB1537330100294')
    # find the lastest timestamp and verify it must be a PASS to update true failure count 
    n_pass_units  = len(df_units[df_units.unit_status=='PASS']['unit_name'].unique())
    n_fail_units  = len(df_units[df_units.unit_status=='FAIL']['unit_name'].unique())
    pass_units    = set(df_units[df_units.unit_status=='PASS']['unit_name'].unique())
    fail_units    = set(df_units[df_units.unit_status=='FAIL']['unit_name'].unique())
    units_overlap = (pass_units & fail_units)
    df_units_overlap = df_units[df_units.unit_name.isin(units_overlap)].sort_values(by='unit_name')
    df_units_overlap = df_units_overlap.groupby('unit_name')[['date_record', 'unit_status']].max()
    n_units_overlap  = df_units_overlap[df_units_overlap.unit_status != 'PASS'].shape[0]
    n_fail_units     = n_fail_units - (len(units_overlap) - n_units_overlap)
    n_pass_units_pct, n_fail_units_pct = n_pass_units/n_units, n_fail_units/n_units
    # create a dict for processing 
    data_metrics = pd.Series({
        'num_units': n_units, 'num_tests': n_tests,
        'num_units_multiple_failures': n_units_mult_failures, 'num_tests_multiple_failures': n_mult_failures,
        'num_pass_tests': n_pass_tests, 'num_fail_tests': n_fail_tests,
        'num_pass_tests_pct': n_pass_tests_pct, 'num_fail_tests_pct': n_fail_tests_pct,
        'num_pass_units': n_pass_units, 'num_fail_units': n_fail_units,
        'num_pass_units_pct': n_pass_units_pct, 'num_fail_units_pct': n_fail_units_pct,
        'num_units_overlapped_passfail': n_units_overlap
    }).sort_values(ascending=False)

    filename = log_path + file_prefix + 'status_metrics.txt'
    write_log(filename, data_metrics, "\nUnit/Experimental Test Metrics:", log=True, format='pretty') 
    return data_metrics

def calc_agg_dated(df_units):
    # date,hourly multi-index
    df_agg_date_hourly = df_units.groupby(['date','hour'])['unit_name'].count()
    df_agg_date_hourly.name = 'units_served'
    df_agg_date_hourly.columns = ['units_served']
    filename = log_path + file_prefix + 'units_served_datehourly.txt'
    write_log(filename, df_agg_date_hourly, format='Pretty') 
    # hourly aggregations   
    df_stats_hourly = df_agg_date_hourly.reset_index()
    df_agg_hourly   = df_stats_hourly.groupby('hour')['units_served'].agg([np.mean, np.median, np.std], axis=1)
    df_agg_hourly = pd.concat( [ df_units.groupby('hour')['unit_name'].count(), df_agg_hourly], axis=1 )
    df_agg_hourly.columns = ['count','average', 'median', 'std']
    filename = log_path + file_prefix + 'units_served_hourly_stats.txt'
    write_log(filename, df_agg_hourly, header=['Count', 'Average', 'Median', 'Std'])
    # hourly summary statistics 
    ds_agg_summary = pd.Series({
         'mean':   df_agg_hourly['count'].mean(), 
         'median': df_agg_hourly['count'].median(), 
         'std':    df_agg_hourly['count'].std()}, name='units_served_hourly')
    filename = log_path + file_prefix + 'units_served_hourly_summary.txt'
    write_log(filename, ds_agg_summary, header=["Units Served Hourly"])
    s = "Units Served Hourly:\nMean: {0:.2f}, Median: {1:.2f}, STD: {2:.2f}"
    print s.format(df_agg_hourly['count'].mean(), df_agg_hourly['count'].median(), df_agg_hourly['count'].std())
    return ds_agg_summary 

def calc_agg_failures(ds, datapath):
    filepath = datapath + ds.unit_status + "/" +  "".join(ds.date.strftime('%Y%m%d')) + "/" 
    filename = filepath + ds.file_name + ds.date_record.strftime('_%Y%m%d_%H%M%S') + '.csv'
    df = pd.read_csv(filename)
    # extract test failures for a given failure and append to 
    df_fail = df[(df.STATUS == 1) | (df.VALUE == 'FAIL')]
    df_test_failures = df_fail.groupby('TEST')['VALUE'].count()
    # keep track of occuring failures 
    return df_test_failures
    
### Configuration Aggregations  
def define_default_configs():
    return [
        {'name': 'voltagedefault', 'prefix': ['V'], 'pattern': ['BOLT', 'PWR']}
    ]

def match(frame, start_cond, pattern_cond):
    # define regex patterns
    pattern_regex = "|".join([p for p in pattern_cond])
    start_regex   = "|".join([p for p in start_cond])
    start_regex   = "^("+ start_regex +")"       
    # create series 
    df_flt = frame[(frame.TEST.str.contains(pattern_regex)) | (frame.TEST.str.contains(start_regex))]
    df_flt = df_flt.reset_index()
    df_flt = df_flt[['TEST','VALUE']].T
    df_flt.columns = [df_flt.ix['TEST']]
    df_flt = df_flt.drop('TEST', axis=0).reset_index().drop('index',axis=1)
    return df_flt

def match_config_patterns(ds, datapath, name, start_cond, pattern_cond): 
    filepath = datapath + ds.unit_status + "/" +  "".join(ds.date.strftime('%Y%m%d')) + "/" 
    filename = filepath + ds.file_name + ds.date_record.strftime('_%Y%m%d_%H%M%S') + '.csv'
    df = pd.read_csv(filename)
    df_patterns = match(df, start_cond, pattern_cond)
    return pd.Series( {k:v.values[0] for k,v in dict(df_patterns).iteritems()} )

def calc_agg_config(frame, datapath, name, start_cond, pattern_cond):
    params = (name, start_cond, pattern_cond)
    df_agg_config = frame.apply(lambda x: match_config_patterns(x, datapath, *params), axis=1).astype('float')
    # calculate aggregations 
    iqr = (df_agg_config.dropna().quantile(0.75, axis=0) - df_agg_config.dropna().quantile(0.25, axis=0))
    df_metric = pd.concat([df_agg_config.mean(axis=0), df_agg_config.median(axis=0), df_agg_config.std(axis=0), 
                           iqr, df_agg_config.min(axis=0), df_agg_config.max(axis=0)], axis=1)
    df_metric.columns = ['mean', 'median', 'std', 'iqr', 'min', 'max']
    df_metric.name = name
    # save to log file 
    filename = log_path + file_prefix + name + '_stats.txt'
    write_log(filename, df_metric, header=["Failure Counts"], format='pretty')
    return df_metric

### Plots/Visualizations  
def plot_units_metrics(metrics, titles):
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,7))
    for data,title,axi in zip(metrics, titles, (ax1,ax2,ax3)):
        sns.barplot(data, data.index, ax=axi)
        axi.set_title(title, fontsize=16, fontweight='bold')
        for tick in axi.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
            tick.label.set_fontweight('bold')
        for tick in axi.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
            tick.label.set_fontweight('bold')            
    fig.set_tight_layout(True)  
    plt.savefig(image_path + file_prefix + 'units_status_metrics.png')

def plot_units_dailyhour(df_units):
    # units per hour tested 
    fig = plt.figure(figsize=(14,6))
    df_units['date'] = df_units.date_record.dt.date
    df_units['hour'] = df_units.date_record.dt.hour
    df_units_dated = df_units.groupby(['date','hour'])['unit_name'].count()
    df_units_dated.unstack(level=0).plot(kind='bar', subplots=False)
    plt.ylabel("Num Units Tested", fontsize=10, fontweight='bold')
    plt.xlabel("Hour", fontsize=10, fontweight='bold')
    plt.title("Distribution per number of units tested", fontsize=13, fontweight='bold')
    fig.set_tight_layout(True)
    plt.savefig(image_path + file_prefix + 'units_tested_datehour.png')    

def plot_units_hourly(df_units):   
    fig = plt.figure(figsize=(14,6)) 
    df_agg_hourly = df_units.groupby(['hour'])['unit_name'].count()
    df_agg_hourly.plot(kind='bar')
    plt.ylabel("Num Units Tested", fontsize=10, fontweight='bold')
    plt.xlabel("Hour", fontsize=10, fontweight='bold')
    plt.title("Hourly Distribution per number of units tested", fontsize=10, fontweight='bold')
    fig.set_tight_layout(True)
    plt.savefig(image_path + file_prefix + 'units_tested_hourly.png')

def plot_failure_metrics(frame):    
    fig = plt.figure(figsize=(14,6))
    sns.barplot(frame, frame.index)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        tick.label.set_fontstyle('italic')
        tick.label.set_fontweight('bold')
    plt.xlabel('Number of Failures', fontsize=10, fontweight='bold')
    plt.title("Failure Test Types Distribution", fontsize=10, fontweight='bold')
    fig.set_tight_layout(True)
    plt.savefig(image_path + file_prefix + 'units_failure_metrics.png')


### Logging 
def write_log(filename, frame, header=None, log=False, format=None):
    if format: 
        with open(filename, 'w') as f: f.write(frame.__repr__())
        if log: print header; print (frame); print
    else:
        frame.to_csv(filename, sep='\t', float_format='%.2f', header=header)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Factory Unit Metrics')
    parser.add_argument('-f', '--fixture', default='fixture', nargs='?', help='default=fixture')
    args = parser.parse_args()
    
    curr_date = "".join( str(datetime.now().date()).split("-") )
    fixture_path  = args.fixture + '/'
    data_path     = fixture_path + 'data/'
    log_path      = fixture_path + 'logs/'   + curr_date + '/'
    image_path    = fixture_path + 'images/' + curr_date + '/'
    config_path   = fixture_path + 'config/'
    file_prefix   = args.fixture.split("/")[-1] + '_'
    root_fixture_path = fixture_path if fixture_path.startswith('/') else os.getcwd() + '/' + fixture_path
    root_data_path    = data_path if fixture_path.startswith('/') else os.getcwd() + '/' + data_path
    # create folder structure if necessary, create tabular dataframe format
    attrs = create_dir_structure()    
    meta_folders = ['logs', 'images', 'config']
    meta_path    = '[' + '|'.join(meta_folders) + ']'
    data_folders = filter(lambda x: x not in meta_folders, attrs['root_dirs'])
    data = [extract_subfolder_data(dir_name, root_fixture_path) for dir_name in attrs['root_dirs']]
    data_dict = {k: v for d in data for k, v in d.items() if not re.compile(meta_path).search(k)}
    df_aggunits = create_units(data_dict, root_data_path)
    # Apply Core Aggregations, Log to Files
    ds_metrics  = calc_agg_stats(df_aggunits).sort_values(ascending=False)
    ds_metrics_summary = calc_agg_dated(df_aggunits)   
    ds_failures = df_aggunits.apply(lambda x: calc_agg_failures(x, data_path), axis=1)
    ds_failures = ds_failures.sum().astype(int)
    ds_failures = ds_failures.drop('OVERALL_TEST_RESULT', axis=0).sort_values(ascending=False)
    filename = log_path + file_prefix + 'testfailuretype_stats.txt'
    write_log(filename, ds_failures[:10], header="\nTop 10 Failure Test Types", log=True, format='pretty')
    # Apply Configuration Aggregations, Log to Files
    if os.path.exists(config_path): 
        with open(config_path + 'config.json') as f:    
            config_json = json.load(f)
            config_tests = config_json['tests']
    else: 
        config_tests = define_default_configs()
    for config in config_tests:
        params = (config['name'], config['prefix'], config['pattern'])
        calc_agg_config(df_aggunits, data_path, *params)
    # Apply Plots 
    ds_metrics_units = ds_metrics.ix[['num_units', 'num_pass_units', 'num_fail_units', 
                                      'num_units_multiple_failures', 'num_units_overlapped_passfail']]
    ds_metrics_tests = ds_metrics.ix[['num_tests', 'num_pass_tests', 
                                      'num_fail_tests','num_tests_multiple_failures']]
    ds_metrics_pct = ds_metrics.ix[['num_pass_units_pct', 'num_pass_tests_pct',
                                    'num_fail_tests_pct', 'num_fail_units_pct']]
    plot_units_metrics((ds_metrics_units, ds_metrics_tests, ds_metrics_pct.sort_values(ascending=False)),
                       ("Unit Metrics", "Pass/Failure Counts", "Pass/Fail Test Percentages"))
    plot_units_dailyhour(df_aggunits)
    plot_units_hourly(df_aggunits)
    plot_failure_metrics(ds_failures)
