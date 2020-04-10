"""
Script for processing VCFv4.2 files
Usage :
    preprocess-dataset.py --input=input file --output=output file --verbose

Arguments:
    --input=input file, -i input file: The input file
    --output=output file, -o output file: The output file
    --verbose, -v : Verbose mode is one, the script is executed providing information about it's actions

"""

from datetime import timedelta
from datetime import datetime
import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
import time
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import csv
from sklearn.preprocessing import OneHotEncoder
from numba import jit


def processing(input_file, output_file, mode):
    logging.info('Using pandas to read the csv file input file')

    """
        Reading the csv file into memory and create a pandas dataframe.
        Very important, the label is read as string, the sample_time and process_creation_time are treated
        like dates, there is a sorting by sample_time and the indexes are recreated after sorting
    """
    _df = pd.read_csv(input_file,
                      sep=",",
                      dtype={'label': np.str},
                      parse_dates=['sample_time', 'process_creation_time']).sort_values(by='sample_time', ascending=True).reset_index(drop=True)

    logging.debug("\n{0}".format(_df))
    logging.info(
        "A dataframe with dimensions {0} was created.".format(_df.shape))
    index = _df.shape[0]
    logging.debug("{0}".format(index))

    """

    b.	Next for the more 120 rows case I found out that the original implementation
    takes the average of all the name and cmd column values in a time slice that
    are the same (for example if the name column has bio set and the cmd column has []
    multiple times then take the average of these rows and turn many to one row,
    if possible let’s do this.

    """
    if mode == 1:
        _df = pd.pivot_table(_df,
                             index=['sample_time', 'process_creation_time',
                                    'status', 'name', 'cmd', 'cwd', 'label'],
                             values=['sample_no', 'exp_no', 'vm_id', 'pid', 'ppid', 'num_threads',
                                     'kb_received', 'kb_sent',
                                     'num_fds',
                                     'cpu_children_sys', 'cpu_children_user',
                                     'cpu_user', 'cpu_sys',
                                     'cpu_percent', 'cpu_num',
                                     'gid_real', 'gid_saved', 'gid_effective',
                                     'mem_swap', 'mem_lib',
                                     'mem_text', 'mem_uss',
                                     'mem_dirty', 'mem_pss',
                                     'mem_shared',
                                     'mem_data', 'mem_vms',
                                     'mem_rss',
                                     'io_write_bytes', 'io_write_chars', 'io_write_count',
                                     'io_read_bytes', 'io_read_chars', 'io_read_count',
                                     'ctx_switches_involuntary', 'ctx_switches_voluntary',
                                     'nice',
                                     'ionice_ioclass', 'ionice_value'],
                             aggfunc=np.median).sort_values(by='sample_time', ascending=True).reset_index()

    """
        Make a pivot table. In the pivot table there are the unique time slices and also the number of records
        in each of the time slice. It is more efficient than making a unique list of the 'sample_time' and
        then counting entries.
    """
    _timestamps = pd.pivot_table(_df,
                                 index=['sample_time'],
                                 values=['status'],
                                 aggfunc=np.count_nonzero).sort_values(by='sample_time', ascending=True).reset_index()

    logging.debug("\n{0}".format(_timestamps))
    logging.info("The number of distinct time slices is {0}".format(
        _timestamps.shape[0]))

    """
        Write a csv with the slices. It is not used any where, it is good to have for troubleshooting
        and understading the efficiency of the calculations
    """
    _timestamps.sort_values(by='sample_time', ascending=True).to_csv(
        output_file + ".slices.csv", sep=';', encoding='utf-8', index=False)

    """
        While the initial description of the dataset says about measurements with 10s of interval,
        this is not the case here. So, everything that is less than 10s apart is merged with
        the previous one only if the total amount of records is less than 120
    """
    logging.info("Reducing the number of time slices")
    _previous_sample_time = None
    _previous_status_count = 0
    j = 0
    for idx, values in _timestamps.iterrows():
        j = j + 1
        if idx == 0:
            _previous_sample_time = values['sample_time']
            _previous_status_count = values['status']
        else:
            if (values['sample_time'] - _previous_sample_time) < timedelta(seconds=10) \
                    and _previous_status_count + values['status'] < 121:
                _df.loc[_df['sample_time'] == values['sample_time'],
                        'sample_time'] = _previous_sample_time
                _previous_status_count = _previous_status_count + \
                    values['status']
            else:
                _previous_sample_time = values['sample_time']
                _previous_status_count = values['status']

    logging.debug("\n{0}".format(_df))

    """
        Again, pivoting in order to have a list of the unique samples times slices and the
        number of record in each one of them
    """
    _timestamps = pd.pivot_table(_df,
                                 index=['sample_time'],
                                 values=['status'],
                                 aggfunc=np.count_nonzero).sort_values(by='sample_time', ascending=True).reset_index()
    logging.debug("\n{0}".format(_timestamps.sort_values(
        by='sample_time', ascending=True)))

    """
        This is the place where major modifications are taking place.
        If the number of entries in a time slice are more than 120, then the indexes are placed into an array
        to be dropped on a later stage.
        If the number of entries in a time slice are less than 120, then records that were marked for deletion
        are being 'recycled' by modifying the sample_time.

        In the case that mode is 1, then:
        a.  First lets track the initial time slice (as we did previously) and if the next time slice
        is not at least 10 seconds apart from the original then we remove that time slice
        (this should take care of those time slices that have very small amounts of
        rows and also not mix data from other time slices).


    """
    _drop_idx = []
    _append_idx = []
    _append_labels = []
    _label = '0'
    _previous_label = '0'
    if mode == 1:
        _previous_sample_time = None
        _previous_status_count = 0
        j = 0
        for idx, values in _timestamps.iterrows():
            j = j + 1
            if idx == 0:
                _previous_sample_time = values['sample_time']
                _previous_status_count = values['status']
            else:
                if (values['sample_time'] - _previous_sample_time) < timedelta(seconds=10):
                    for i in _df.loc[_df['sample_time'] == values['sample_time']].index:
                        _drop_idx.append(i)
                else:
                    _previous_sample_time = values['sample_time']
                    _previous_status_count = values['status']

    logging.info("Dropping the entries")
    _df.drop(_drop_idx, inplace=True)

    _timestamps = pd.pivot_table(_df,
                                 index=['sample_time'],
                                 values=['status'],
                                 aggfunc=np.count_nonzero).sort_values(by='sample_time', ascending=True).reset_index()
    logging.debug("\n{0}".format(_timestamps.sort_values(
        by='sample_time', ascending=True)))

    for idx, values in _timestamps.iterrows():
        logging.debug("Processing {0:6d}, status is {1:4d}, time step {2}".format(
            idx, values['status'], values['sample_time']))
        if values['status'] > 120:
            _label = _df.loc[_df['sample_time'] ==
                             values['sample_time'], 'label'].values[0]

            if _label != _previous_label:
                if _drop_idx:
                    logging.info("Dropping the entries")
                    _df.drop(_drop_idx, inplace=True)
                    _drop_idx = []

            y = 0
            for j in _df.loc[_df['sample_time'] == values['sample_time']].index:
                if y > 119:
                    _drop_idx.append(j)
                y = y + 1
            _previous_label = _label
        elif values['status'] < 120:
            _label = _df.loc[_df['sample_time'] ==
                             values['sample_time'], 'label'].values[0]

            if _label != _previous_label:
                if _drop_idx:
                    logging.info("Dropping the entries")
                    _df.drop(_drop_idx, inplace=True)
                    _drop_idx = []

            for i in range(values['status'], 120):
                if _drop_idx:
                    j = _drop_idx.pop()
                    _df.loc[j, 'sample_time'] = values['sample_time']
                else:
                    _append_idx.append(values['sample_time'])
                    _append_labels.append(_label)
            _previous_label = _label
        else:
            _label = _df.loc[_df['sample_time'] ==
                             values['sample_time'], 'label'].values[0]

            if _label != _previous_label:
                if _drop_idx:
                    logging.info("Dropping the entries")
                    _df.drop(_drop_idx, inplace=True)
                    _drop_idx = []

            _previous_label = _label

    """
        Using the list of indexes and sample_time to make an array of new records.
        The median is used here instead of 0 and for the categorical columns the value 'missing'
    """
    logging.info("Append to the list")
    _append_list = []
    previous_v = None
    _sample_no = 0
    _exp_no = 0
    _vm_id = 0
    _pid = 0
    _ppid = 0
    _num_threads = 0
    _kb_received = 0
    _kb_sent = 0
    _num_fds = 0
    _cpu_children_sys = 0
    _cpu_children_user = 0
    _cpu_user = 0
    _cpu_sys = 0
    _cpu_percent = 0
    _cpu_num = 0
    _gid_real = 0
    _gid_saved = 0
    _gid_effective = 0
    _mem_swap = 0
    _mem_lib = 0
    _mem_text = 0
    _mem_uss = 0
    _mem_dirty = 0
    _mem_pss = 0
    _mem_shared = 0
    _mem_dirty = 0
    _mem_data = 0
    _mem_vms = 0
    _mem_rss = 0
    _io_write_bytes = 0
    _io_write_chars = 0
    _io_write_count = 0
    _io_read_bytes = 0
    _io_read_chars = 0
    _io_read_count = 0
    _ctx_switches_involuntary = 0
    _ctx_switches_voluntary = 0
    _nice = 0
    _ionice_ioclass = 0
    _ionice_value = 0

    _medians = pd.pivot_table(_df,
                              index=['sample_time'],
                              values=['sample_no', 'exp_no', 'vm_id', 'pid', 'ppid', 'num_threads',
                                      'kb_received', 'kb_sent',
                                      'num_fds',
                                      'cpu_children_sys', 'cpu_children_user',
                                      'cpu_user', 'cpu_sys',
                                      'cpu_percent', 'cpu_num',
                                      'gid_real', 'gid_saved', 'gid_effective',
                                      'mem_swap', 'mem_lib',
                                      'mem_text', 'mem_uss',
                                      'mem_dirty', 'mem_pss',
                                      'mem_shared',
                                      'mem_data', 'mem_vms',
                                      'mem_rss',
                                      'io_write_bytes', 'io_write_chars', 'io_write_count',
                                      'io_read_bytes', 'io_read_chars', 'io_read_count',
                                      'ctx_switches_involuntary', 'ctx_switches_voluntary',
                                      'nice',
                                      'ionice_ioclass', 'ionice_value'],
                              aggfunc=np.median).sort_values(by='sample_time', ascending=True).reset_index()

    logging.debug("\n{0}".format(_medians))

    for v, j in zip(_append_idx, _append_labels):
        logging.debug("Processing {0}, {1}".format(v, j))
        if previous_v and previous_v == v:
            pass
        else:
            _sample_no = _medians.loc[_medians.sample_time ==
                                      v, 'sample_no'].values[0]
            logging.debug("\nSample No Value {0}".format(_sample_no))
            _exp_no = _medians.loc[_medians.sample_time ==
                                   v, 'exp_no'].values[0]
            _vm_id = _medians.loc[_medians.sample_time == v, 'vm_id'].values[0]
            _pid = _medians.loc[_medians.sample_time == v, 'pid'].values[0]
            _ppid = _medians.loc[_medians.sample_time == v, 'ppid'].values[0]
            _num_threads = _medians.loc[_medians.sample_time ==
                                        v, 'num_threads'].values[0]
            _kb_received = _medians.loc[_medians.sample_time ==
                                        v, 'kb_received'].values[0]
            _kb_sent = _medians.loc[_medians.sample_time ==
                                    v, 'kb_sent'].values[0]
            _num_fds = _medians.loc[_medians.sample_time ==
                                    v, 'num_fds'].values[0]
            _cpu_children_sys = _medians.loc[_medians.sample_time ==
                                             v, 'cpu_children_sys'].values[0]
            _cpu_children_user = _medians.loc[_medians.sample_time ==
                                              v, 'cpu_children_user'].values[0]
            _cpu_user = _medians.loc[_medians.sample_time ==
                                     v, 'cpu_user'].values[0]
            _cpu_sys = _medians.loc[_medians.sample_time ==
                                    v, 'cpu_sys'].values[0]
            _cpu_percent = _medians.loc[_medians.sample_time ==
                                        v, 'cpu_percent'].values[0]
            _cpu_num = _medians.loc[_medians.sample_time ==
                                    v, 'cpu_num'].values[0]
            _gid_real = _medians.loc[_medians.sample_time ==
                                     v, 'gid_real'].values[0]
            _gid_saved = _medians.loc[_medians.sample_time ==
                                      v, 'gid_saved'].values[0]
            _gid_effective = _medians.loc[_medians.sample_time ==
                                          v, 'gid_effective'].values[0]
            _mem_swap = _medians.loc[_medians.sample_time ==
                                     v, 'mem_swap'].values[0]
            _mem_lib = _medians.loc[_medians.sample_time ==
                                    v, 'mem_lib'].values[0]
            _mem_text = _medians.loc[_medians.sample_time ==
                                     v, 'mem_text'].values[0]
            _mem_uss = _medians.loc[_medians.sample_time ==
                                    v, 'mem_uss'].values[0]
            _mem_dirty = _medians.loc[_medians.sample_time ==
                                      v, 'mem_dirty'].values[0]
            _mem_pss = _medians.loc[_medians.sample_time ==
                                    v, 'mem_pss'].values[0]
            logging.debug("\nmem_pss {0}".format(_mem_pss))
            _mem_shared = _medians.loc[_medians.sample_time ==
                                       v, 'mem_shared'].values[0]
            _mem_data = _medians.loc[_medians.sample_time ==
                                     v, 'mem_data'].values[0]
            _mem_vms = _medians.loc[_medians.sample_time ==
                                    v, 'mem_vms'].values[0]
            _mem_rss = _medians.loc[_medians.sample_time ==
                                    v, 'mem_rss'].values[0]
            _io_write_bytes = _medians.loc[_medians.sample_time ==
                                           v, 'io_write_bytes'].values[0]
            _io_write_chars = _medians.loc[_medians.sample_time ==
                                           v, 'io_write_chars'].values[0]
            _io_write_count = _medians.loc[_medians.sample_time ==
                                           v, 'io_write_count'].values[0]
            _io_read_bytes = _medians.loc[_medians.sample_time ==
                                          v, 'io_read_bytes'].values[0]
            _io_read_chars = _medians.loc[_medians.sample_time ==
                                          v, 'io_read_chars'].values[0]
            _io_read_count = _medians.loc[_medians.sample_time ==
                                          v, 'io_read_count'].values[0]
            _ctx_switches_involuntary = _medians.loc[_medians.sample_time ==
                                                     v, 'ctx_switches_involuntary'].values[0]
            _ctx_switches_voluntary = _medians.loc[_medians.sample_time ==
                                                   v, 'ctx_switches_voluntary'].values[0]
            _nice = _medians.loc[_medians.sample_time == v, 'nice'].values[0]
            _ionice_ioclass = _medians.loc[_medians.sample_time ==
                                           v, 'ionice_ioclass'].values[0]
            _ionice_value = _medians.loc[_medians.sample_time ==
                                         v, 'ionice_value'].values[0]

        _append_list.append([_sample_no,
                             _exp_no,
                             _vm_id,
                             _pid,
                             _ppid,
                             v,
                             v,
                             'missing',
                             _num_threads,
                             _kb_received,
                             _kb_sent,
                             _num_fds,
                             _cpu_children_sys,
                             _cpu_children_user,
                             _cpu_user,
                             _cpu_sys,
                             _cpu_percent,
                             _cpu_num,
                             'missing',
                             _gid_real,
                             _gid_saved,
                             _gid_effective,
                             _mem_swap,
                             _mem_lib,
                             _mem_text,
                             _mem_uss,
                             _mem_dirty,
                             _mem_pss,
                             _mem_shared,
                             _mem_data,
                             _mem_vms,
                             _mem_rss,
                             _io_write_bytes,
                             _io_write_chars,
                             _io_write_count,
                             _io_read_bytes,
                             _io_read_chars,
                             _io_read_count,
                             _ctx_switches_involuntary,
                             _ctx_switches_voluntary,
                             _nice,
                             _ionice_ioclass,
                             _ionice_value,
                             'missing',
                             'missing',
                             j])
        previous_v = v

    """
        Convert the array into a dataframe
    """

    logging.info('Create dataframe')

    _append_dataframe = pd.DataFrame(data=_append_list,
                                     columns=['sample_no',
                                              'exp_no',
                                              'vm_id',
                                              'pid',
                                              'ppid',
                                              'sample_time',
                                              'process_creation_time',
                                              'status',
                                              'num_threads',
                                              'kb_received',
                                              'kb_sent',
                                              'num_fds',
                                              'cpu_children_sys',
                                              'cpu_children_user',
                                              'cpu_user',
                                              'cpu_sys',
                                              'cpu_percent',
                                              'cpu_num',
                                              'name',
                                              'gid_real',
                                              'gid_saved',
                                              'gid_effective',
                                              'mem_swap',
                                              'mem_lib',
                                              'mem_text',
                                              'mem_uss',
                                              'mem_dirty',
                                              'mem_pss',
                                              'mem_shared',
                                              'mem_data',
                                              'mem_vms',
                                              'mem_rss',
                                              'io_write_bytes',
                                              'io_write_chars',
                                              'io_write_count',
                                              'io_read_bytes',
                                              'io_read_chars',
                                              'io_read_count',
                                              'ctx_switches_involuntary',
                                              'ctx_switches_voluntary',
                                              'nice',
                                              'ionice_ioclass',
                                              'ionice_value',
                                              'cmd',
                                              'cwd',
                                              'label'])

    """
        Dropping the indexes that are not required any more and the relevant rows.
    """

    logging.info("Dropping the entries")
    _df.drop(_drop_idx, inplace=True)

    """
        Append the padding rows
    """

    logging.info("Appending the padding")
    logging.debug("\n{0}".format(_append_dataframe))

    """
        Sort and reset indexes
    """
    _df1 = _df.append(_append_dataframe, sort=False).reset_index(
        drop=True).sort_values(by='sample_time', ascending=True).copy(deep=True)
    logging.debug("\n{0}".format(_df1))

    _timestamps = pd.pivot_table(_df1,
                                 index=['sample_time'],
                                 values=['status'],
                                 aggfunc=np.count_nonzero).sort_values(by='sample_time', ascending=True)
    logging.debug("\n{0}".format(_timestamps))

    """
        Drop columns
    """
    cols = ['name',
            'cmd']
    for v in cols:
        logging.info("Drop the column {0}".format(v))
        _df1.drop([v], axis=1, inplace=True)

    """
        Encoding
    """
    cols = ['status',
            'cwd']
    for v in cols:
        _df1[v] = _df1[v].fillna('missing')
        logging.info(
            "Apply one hot encoder with pandas get_dummies {0}".format(v))
        _df1 = pd.get_dummies(_df1, columns=[v])

    cols = ['sample_no',
            'exp_no',
            'vm_id',
            'pid',
            'ppid',
            'process_creation_time']
    for v in cols:
        logging.info("Drop the column {0}".format(v))
        _df1.drop([v], axis=1, inplace=True)

    cols = ['num_threads',
            'kb_received',
            'kb_sent',
            'num_fds',
            'cpu_children_sys',
            'cpu_children_user',
            'cpu_user',
            'cpu_sys',
            'cpu_percent',
            'cpu_num',
            'gid_real',
            'gid_saved',
            'gid_effective',
            'mem_swap',
            'mem_lib',
            'mem_text',
            'mem_uss',
            'mem_dirty',
            'mem_pss',
            'mem_shared',
            'mem_data',
            'mem_vms',
            'mem_rss',
            'io_write_bytes',
            'io_write_chars',
            'io_write_count',
            'io_read_bytes',
            'io_read_chars',
            'io_read_count',
            'ctx_switches_involuntary',
            'ctx_switches_voluntary',
            'nice',
            'ionice_ioclass',
            'ionice_value']

    scaler = StandardScaler()
    for v in cols:
        logging.info("Apply scalling to column {0}".format(v))
        logging.debug("\n{0}".format(_df1[v]))
        _df1[v] = scaler.fit_transform(
            _df1[v].astype(float).values.reshape(-1, 1))

    cols = ['label']
    for v in cols:
        logging.info("Do nothing with column {0}".format(v))

    logging.debug("\n{0}".format(_df1))

    _df1.to_csv(output_file, sep=',', encoding='utf-8',
                index=False, quoting=csv.QUOTE_ALL)

    _tf = pd.DataFrame(columns=['X', 'y'])
    _TimeUniqueValues = list(set(_df1['sample_time']))
    _TimeUniqueValues.sort()

    for idx, sample_time in enumerate(_TimeUniqueValues):
        data = _df1.loc[_df1['sample_time'] == sample_time]
        X = data.iloc[:, 1:-1].values
        y = math.ceil(sum(data.iloc[:, -1].values) /
                      len(data.iloc[:, -1].values))
        _tf.loc[idx] = [X, y]

    # Final Variables
    X = _tf.iloc[:, 0].values
    y = _tf.iloc[:, 1].values

    # Test Train Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Saving them to feed model
    np.save("Dataset/Processed/X_train", X_train)
    np.save("Dataset/Processed/X_test", X_test)
    np.save("Dataset/Processed/y_train", y_train)
    np.save("Dataset/Processed/y_test", y_test)


def validations(arguments=None):
    if arguments.input is not None and arguments.output is not None:
        v_input_file = os.path.abspath(os.path.realpath(arguments.input))
        v_output_file = os.path.abspath(os.path.realpath(arguments.output))
        v_mode = arguments.mode

        if v_input_file == v_output_file:
            logging.info("The input {0} and output {1} files cannot be the same".format(v_input_file,
                                                                                        v_output_file))
            sys.exit(1)

        logging.debug("input file is {0}".format(v_input_file))
        logging.debug("output file is {0}".format(v_output_file))

        if not os.path.isfile(v_input_file):
            logging.critical("file {0} does not exist...".format(v_input_file))
            sys.exit(1)

        if os.path.isfile(v_output_file):
            logging.warning("directory {0} exist...".format(v_output_file))
        return v_input_file, v_output_file, v_mode
    else:
        logging.info("input, output files need to be defined")
        sys.exit(1)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_mutually_exclusive_group(required=False)
    PARSER.add_argument('-i', '--input',
                        type=str,
                        action="store",
                        dest="input",
                        help='The source directory')
    PARSER.add_argument('-o', '--output',
                        type=str,
                        action="store",
                        dest="output",
                        help='The output directory')
    PARSER.add_argument('-v', '--verbose',
                        dest="verbose",
                        action="store_true",
                        help='display messages while execution')
    PARSER.add_argument('-m', '--mode',
                        type=str,
                        action="store",
                        dest="mode",
                        help='introduce new functonality')

    PARSER.set_defaults(verbose=False)

    t1 = time.time()

    try:
        ARGS = PARSER.parse_args()
    except IndexError as ex:
        print("Usage python preprocess-dataset.py --input='input file' --output='output file' --verbose --mode=1")
        sys.exit(1)

    logging.basicConfig(level=(logging.DEBUG if ARGS.verbose else logging.INFO),
                        format='%(asctime)s : %(levelname)-12s : %(message)s')

    logging.debug("The script is invoked with the following parameters")
    logging.debug("verbose is {0}".format(ARGS.verbose))
    logging.debug("mode is {0}".format(ARGS.mode))
    logging.debug("The input file is {0}".format(ARGS.input))
    logging.debug("The output file is {0}".format(ARGS.output))

    input_file, output_file, mode = validations(ARGS)

    processing(input_file, output_file, mode)
    t2 = time.time()
    logging.info("Execution time {0}".format(t2 - t1))
