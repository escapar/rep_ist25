import os
import numpy as np
import gc
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
random.seed(42)
np.random.seed(42)

def find_max_individual_token(folder):
    max_value = 0
    for (root, dirs, files) in os.walk(folder):
        for file in files:
            if file.startswith('.'):
                continue
            filepath = os.path.join(root, file)
            input_str = open(filepath, 'r', errors='ignore').read()
            input_str = input_str.replace('\\t', ' ')
            np_arr = np.fromstring(input_str, dtype=np.int32, sep=' ')
            for item in np_arr:
                if item > max_value:
                    max_value = item
                if item < 0:
                    if os.path.exists(filepath):
                        os.remove(filepath)
    return max_value

def find_input_size(folder, max_width, max_height):
    for (root, dirs, files) in os.walk(folder):
        for file in files:
            cur_height = 0
            with open(os.path.join(root, file), encoding='utf8') as fp:
                for line in fp:
                    cur_height += 1
                    cur_width = len(line)
                    if cur_height > max_height:
                        max_height = cur_height
                    if cur_width > max_width:
                        max_width = cur_width
    return (max_width, max_height)

def find_input_size_1d(folder, max_width):
    for (root, dirs, files) in os.walk(folder):
        for file in files:
            cur_width = os.path.getsize(os.path.join(root, file))
            if cur_width > max_width:
                max_width = cur_width
    return max_width

def find_tokenized_input_size_1d(folder, max_width):
    for (root, dirs, files) in os.walk(folder):
        for file in files:
            if file.startswith('.'):
                continue
            input_str = open(os.path.join(root, file), 'r', errors='ignore').read()
            input_str = input_str.replace('\\t', ' ')
            np_arr = np.fromstring(input_str, dtype=np.int32, sep=' ')
            cur_width = len(np_arr)
            if cur_width > max_width:
                max_width = cur_width
    return max_width

def tokenized_input_size_1d_all(pos_folder, neg_folder):
    max_width = 0
    max_width = find_tokenized_input_size_1d(pos_folder, max_width)
    max_width = find_tokenized_input_size_1d(neg_folder, max_width)
    return max_width

def item_line_count(path):
    if isfile(path):
        return len(open(path, 'rb').readlines())
    else:
        return 0

def get_total_cases_2d(folder_path):
    count = 0
    for file in os.listdir(folder_path):
        filepath = os.path.join(folder_path, file)
        if os.path.isfile(filepath):
            with open(filepath, 'r', errors='ignore') as f:
                for line in f:
                    if len(line) < 2:
                        count += 1
    return count

def get_total_cases(folder_path):
    return sum(map(lambda item: item_line_count(join(folder_path, item)), listdir(folder_path)))

def get_data_balanced(data_path, train_validate_ratio=0.7, max_training_samples=5000, is_final=False):
    max_input_length = get_outlier_threshold(data_path, z=1)
    folder_path = os.path.join(data_path, 'Positive')
    pos_data_arr = _retrieve_data(folder_path, max_input_length)
    shuffle(pos_data_arr, random_state=42)
    total_positive_cases = len(pos_data_arr)
    total_training_positive_cases = int(train_validate_ratio * total_positive_cases)
    total_eval_positive_cases = int(total_positive_cases - total_training_positive_cases)
    if max_training_samples is None:
        max_training_samples = total_positive_cases
    if max_eval_samples is None:
        max_eval_samples = total_eval_positive_cases + total_eval_negative_cases
    if total_training_positive_cases > max_training_samples:
        total_training_positive_cases = max_training_samples
    folder_path = os.path.join(data_path, 'Negative')
    neg_data_arr = _retrieve_data(folder_path, max_input_length)
    shuffle(neg_data_arr, random_state=42)
    total_negative_cases = len(neg_data_arr)
    total_training_negative_cases = int(train_validate_ratio * total_negative_cases)
    total_eval_negative_cases = int(total_negative_cases - total_training_negative_cases)
    training_data = []
    training_data.extend(pos_data_arr[0:total_training_positive_cases])
    training_data.extend(neg_data_arr[0:total_training_negative_cases])
    training_data_arr = np.array(training_data, dtype=np.float32)
    training_labels = np.empty(shape=[len(training_data_arr)], dtype=np.float32)
    training_labels[0:total_training_positive_cases] = 1.0
    training_labels[total_training_positive_cases:len(training_data_arr)] = 0.0
    eval_data = []
    eval_data.extend(pos_data_arr[len(pos_data_arr) - total_eval_positive_cases:])
    eval_data.extend(neg_data_arr[len(neg_data_arr) - total_eval_negative_cases:])
    eval_data_arr = np.array(eval_data, dtype=np.float32)
    eval_labels = np.empty(shape=[len(eval_data_arr)], dtype=np.float32)
    eval_labels[0:total_eval_positive_cases] = 1.0
    eval_labels[total_eval_positive_cases:] = 0.0
    training_data = training_data_arr.reshape((len(training_labels), max_input_length, 1))
    eval_data = eval_data_arr.reshape((len(eval_labels), max_input_length, 1))
    (training_data, training_labels) = shuffle(training_data, training_labels)
    (eval_data, eval_labels) = shuffle(eval_data, eval_labels)
    return (training_data, training_labels, eval_data, eval_labels, max_input_length)

def get_data_autoencoder(data_path, train_validate_ratio=0.7, max_training_samples=5000, max_eval_samples=150000):
    gc.collect()
    max_input_length = get_outlier_threshold(data_path, z=1)
    all_inputs = []
    folder_path = os.path.join(data_path, 'Positive')
    pos_data_arr = _retrieve_data(folder_path, max_input_length)
    shuffle(pos_data_arr, random_state=42)
    total_positive_cases = len(pos_data_arr)
    total_eval_positive_cases = total_positive_cases
    folder_path = os.path.join(data_path, 'Negative')
    neg_data_arr = _retrieve_data(folder_path, max_input_length)
    shuffle(neg_data_arr, random_state=42)
    total_negative_cases = len(neg_data_arr)
    total_training_negative_cases = int(train_validate_ratio * total_negative_cases)
    total_eval_negative_cases = int(total_negative_cases - total_training_negative_cases)
    total_training_negative_cases = min(max_training_samples, total_training_negative_cases)
    training_data = []
    training_data.extend(neg_data_arr[0:total_training_negative_cases])
    training_data_arr = np.array(training_data, dtype=np.float32)
    if total_eval_negative_cases > max_eval_samples:
        removed_sample_percent = (total_eval_negative_cases - max_eval_samples) / total_eval_negative_cases
        total_eval_positive_cases = int(total_eval_positive_cases - total_eval_positive_cases * removed_sample_percent)
        total_eval_negative_cases = max_eval_samples
    eval_data = []
    eval_data.extend(pos_data_arr[len(pos_data_arr) - total_eval_positive_cases:])
    eval_data.extend(neg_data_arr[len(neg_data_arr) - total_eval_negative_cases:])
    eval_data_arr = np.array(eval_data, dtype=np.float32)
    eval_labels = np.empty(shape=[len(eval_data_arr)], dtype=np.float32)
    eval_labels[0:total_eval_positive_cases] = 1.0
    eval_labels[total_eval_positive_cases:] = 0.0
    training_data = training_data_arr.reshape((len(training_data_arr), max_input_length))
    eval_data = eval_data_arr.reshape((len(eval_labels), max_input_length))
    training_data = shuffle(training_data)
    (eval_data, eval_labels) = shuffle(eval_data, eval_labels)
    return (training_data, eval_data, eval_labels, max_input_length)

def get_data(data_path, train_validate_ratio=0.7, max_training_samples=None, max_eval_samples=None, is_c2v=False):
    gc.collect()
    max_input_length = get_outlier_threshold(data_path, z=1, is_c2v=is_c2v)
    all_inputs = []
    folder_path = os.path.join(data_path, 'Positive')
    pos_data_arr = _retrieve_data(folder_path, max_input_length, is_c2v)
    shuffle(pos_data_arr, random_state=42)
    total_positive_cases = len(pos_data_arr)
    total_training_positive_cases = int(train_validate_ratio * total_positive_cases)
    total_eval_positive_cases = int(total_positive_cases - total_training_positive_cases)
    folder_path = os.path.join(data_path, 'Negative')
    neg_data_arr = _retrieve_data(folder_path, max_input_length, is_c2v)
    shuffle(neg_data_arr, random_state=42)
    total_negative_cases = len(neg_data_arr)
    total_training_negative_cases = int(train_validate_ratio * total_negative_cases)
    total_eval_negative_cases = int(total_negative_cases - total_training_negative_cases)
    total_training_positive_cases = total_training_negative_cases = min(max_training_samples, min(total_training_positive_cases, total_training_negative_cases))
    training_data = []
    training_data.extend(pos_data_arr[0:total_training_positive_cases])
    training_data.extend(neg_data_arr[0:total_training_negative_cases])
    training_data_arr = np.array(training_data, dtype=np.float32)
    training_labels = np.empty(shape=[len(training_data_arr)], dtype=np.float32)
    training_labels[0:total_training_positive_cases] = 1.0
    training_labels[total_training_positive_cases:len(training_data_arr)] = 0.0
    if total_eval_negative_cases > max_eval_samples:
        removed_sample_percent = (total_eval_negative_cases - max_eval_samples) / total_eval_negative_cases
        total_eval_positive_cases = int(total_eval_positive_cases - total_eval_positive_cases * removed_sample_percent)
        total_eval_negative_cases = max_eval_samples
    eval_data = []
    eval_data.extend(pos_data_arr[len(pos_data_arr) - total_eval_positive_cases:])
    eval_data.extend(neg_data_arr[len(neg_data_arr) - total_eval_negative_cases:])
    eval_data_arr = np.array(eval_data, dtype=np.float32)
    eval_labels = np.empty(shape=[len(eval_data_arr)], dtype=np.float32)
    eval_labels[0:total_eval_positive_cases] = 1.0
    eval_labels[total_eval_positive_cases:] = 0.0
    training_data = training_data_arr.reshape((len(training_labels), max_input_length, 1))
    eval_data = eval_data_arr.reshape((len(eval_labels), max_input_length, 1))
    (training_data, training_labels) = shuffle(training_data, training_labels)
    (eval_data, eval_labels) = shuffle(eval_data, eval_labels)
    return (training_data, training_labels, eval_data, eval_labels, max_input_length)

def get_data_rq2(training_data_path, eval_data_path, out_folder, case_string, train_validate_ratio=0.7, max_training_samples=5000, max_eval_samples=150000):
    gc.collect()
    max_input_length = get_outlier_threshold(training_data_path, z=1)
    tr_pos_data_arr = _retrieve_data(os.path.join(training_data_path, 'Positive'), max_input_length)
    total_tr_positive_cases = len(tr_pos_data_arr)
    total_training_positive_cases = int(train_validate_ratio * total_tr_positive_cases)
    ev_pos_data_arr = _retrieve_data(os.path.join(eval_data_path, 'Positive'), max_input_length)
    total_eval_positive_cases = len(ev_pos_data_arr)
    tr_neg_data_arr = _retrieve_data(os.path.join(training_data_path, 'Negative'), max_input_length)
    total_tr_negative_cases = len(tr_neg_data_arr)
    total_training_negative_cases = int(train_validate_ratio * total_tr_negative_cases)
    ev_neg_data_arr = _retrieve_data(os.path.join(eval_data_path, 'Negative'), max_input_length)
    total_eval_negative_cases = len(ev_neg_data_arr)
    total_training_positive_cases = total_training_negative_cases = min(max_training_samples, min(total_training_positive_cases, total_training_negative_cases))
    training_data = []
    training_data.extend(tr_pos_data_arr[0:total_training_positive_cases])
    training_data.extend(tr_neg_data_arr[0:total_training_negative_cases])
    training_data_arr = np.array(training_data, dtype=np.float32)
    if total_eval_negative_cases > max_eval_samples:
        removed_sample_percent = (total_eval_negative_cases - max_eval_samples) / total_eval_negative_cases
        total_eval_positive_cases = int(total_eval_positive_cases - total_eval_positive_cases * removed_sample_percent)
        total_eval_negative_cases = max_eval_samples
    eval_data = []
    eval_data.extend(ev_pos_data_arr[0:total_eval_positive_cases])
    eval_data.extend(ev_neg_data_arr[0:total_eval_negative_cases])
    eval_data_arr = np.array(eval_data, dtype=np.float32)
    training_labels = np.empty(shape=[len(training_data_arr)], dtype=np.float32)
    training_labels[0:total_training_positive_cases] = 1.0
    training_labels[total_training_positive_cases:len(training_data_arr)] = 0.0
    eval_labels = np.empty(shape=[len(eval_data_arr)], dtype=np.float32)
    eval_labels[0:total_eval_positive_cases] = 1.0
    eval_labels[total_eval_positive_cases:len(eval_data_arr)] = 0.0
    training_data_arr = training_data_arr.reshape((len(training_labels), max_input_length, 1))
    eval_data_arr = eval_data_arr.reshape((len(eval_labels), max_input_length, 1))
    (training_data_arr, training_labels) = shuffle(training_data_arr, training_labels)
    (eval_data_arr, eval_labels) = shuffle(eval_data_arr, eval_labels)
    write_input_data_summary(out_folder, case_string, total_training_positive_cases, total_training_negative_cases, total_eval_positive_cases, total_eval_negative_cases)
    return (training_data_arr, training_labels, eval_data_arr, eval_labels, max_input_length)

def get_data_2d_rq2(training_data_path, eval_data_path, out_folder, case_string, train_validate_ratio=0.7, max_training_samples=5000):
    gc.collect()
    (max_input_width, max_input_height) = get_outlier_threshold_2d(training_data_path, z=1)
    tr_pos_data_arr = _retrieve_data_2d(os.path.join(training_data_path, 'Positive'), max_input_width, max_input_height)
    total_tr_positive_cases = len(tr_pos_data_arr)
    total_training_positive_cases = int(train_validate_ratio * total_tr_positive_cases)
    ev_pos_data_arr = _retrieve_data_2d(os.path.join(eval_data_path, 'Positive'), max_input_width, max_input_height)
    total_eval_positive_cases = len(ev_pos_data_arr)
    tr_neg_data_arr = _retrieve_data_2d(os.path.join(training_data_path, 'Negative'), max_input_width, max_input_height)
    total_tr_negative_cases = len(tr_neg_data_arr)
    total_training_negative_cases = int(train_validate_ratio * total_tr_negative_cases)
    ev_neg_data_arr = _retrieve_data_2d(os.path.join(eval_data_path, 'Negative'), max_input_width, max_input_height)
    total_eval_negative_cases = len(ev_neg_data_arr)
    total_training_positive_cases = total_training_negative_cases = min(max_training_samples, min(total_training_positive_cases, total_training_negative_cases))
    training_data = []
    training_data.extend(tr_pos_data_arr[0:total_training_positive_cases])
    training_data.extend(tr_neg_data_arr[0:total_training_negative_cases])
    training_data_arr = np.array(training_data, dtype=np.float32)
    eval_data = []
    eval_data.extend(ev_pos_data_arr[0:total_eval_positive_cases])
    eval_data.extend(ev_neg_data_arr[0:total_eval_negative_cases])
    eval_data_arr = np.array(eval_data, dtype=np.float32)
    training_labels = np.empty(shape=[len(training_data_arr)], dtype=np.float32)
    training_labels[0:total_training_positive_cases] = 1.0
    training_labels[total_training_positive_cases:len(training_data_arr)] = 0.0
    eval_labels = np.empty(shape=[len(eval_data_arr)], dtype=np.float32)
    eval_labels[0:total_eval_positive_cases] = 1.0
    eval_labels[total_eval_positive_cases:len(eval_data_arr)] = 0.0
    training_data_arr = training_data_arr.reshape((len(training_labels), max_input_height, max_input_width, 1))
    eval_data_arr = eval_data_arr.reshape((len(eval_labels), max_input_height, max_input_width, 1))
    (training_data_arr, training_labels) = shuffle(training_data_arr, training_labels)
    (eval_data_arr, eval_labels) = shuffle(eval_data_arr, eval_labels)
    write_input_data_summary(out_folder, case_string, total_training_positive_cases, total_training_negative_cases, total_eval_positive_cases, total_eval_negative_cases)
    return (training_data_arr, training_labels, eval_data_arr, eval_labels, max_input_height, max_input_width)

def _retrieve_data(path, max_len, is_c2v=False):
    input = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r', errors='ignore') as file_read:
            for line in file_read:
                input_str = line.replace('\t', ' ')
                if is_c2v:
                    arr = np.fromstring(input_str, dtype=np.float, sep=' ', count=max_len)
                    arr_size = len(np.fromstring(input_str, dtype=np.float, sep=' '))
                else:
                    arr = np.fromstring(input_str, dtype=np.int32, sep=' ', count=max_len)
                    arr_size = len(np.fromstring(input_str, dtype=np.int32, sep=' '))
                if arr_size <= max_len:
                    arr[arr_size:max_len] = 0
                    input.append(arr)
    return input

def get_data_2d(data_path, out_folder, case_string, train_validate_ratio=0.7, max_training_samples=5000):
    gc.collect()
    (max_input_width, max_input_height) = get_outlier_threshold_2d(data_path, z=1)
    all_inputs = []
    folder_path = os.path.join(data_path, 'Positive')
    pos_data_arr = _retrieve_data_2d(folder_path, max_input_width, max_input_height)
    shuffle(pos_data_arr, random_state=42)
    total_positive_cases = len(pos_data_arr)
    total_training_positive_cases = int(train_validate_ratio * total_positive_cases)
    total_eval_positive_cases = int(total_positive_cases - total_training_positive_cases)
    folder_path = os.path.join(data_path, 'Negative')
    neg_data_arr = _retrieve_data_2d(folder_path, max_input_width, max_input_height)
    shuffle(neg_data_arr, random_state=42)
    total_negative_cases = len(neg_data_arr)
    total_training_negative_cases = int(train_validate_ratio * total_negative_cases)
    total_eval_negative_cases = int(total_negative_cases - total_training_negative_cases)
    total_training_positive_cases = total_training_negative_cases = min(max_training_samples, min(total_training_positive_cases, total_training_negative_cases))
    training_data = []
    training_data.extend(pos_data_arr[0:total_training_positive_cases])
    training_data.extend(neg_data_arr[0:total_training_negative_cases])
    training_data_arr = np.array(training_data, dtype=np.float32)
    training_labels = np.empty(shape=[len(training_data_arr)], dtype=np.float32)
    training_labels[0:total_training_positive_cases] = 1.0
    training_labels[total_training_positive_cases:len(training_data_arr)] = 0.0
    eval_data = []
    eval_data.extend(pos_data_arr[len(pos_data_arr) - total_eval_positive_cases:])
    eval_data.extend(neg_data_arr[len(neg_data_arr) - total_eval_negative_cases:])
    eval_data_arr = np.array(eval_data, dtype=np.float32)
    eval_labels = np.empty(shape=[len(eval_data_arr)], dtype=np.float32)
    eval_labels[0:total_eval_positive_cases] = 1.0
    eval_labels[total_eval_positive_cases:] = 0.0
    write_input_data_summary(out_folder, case_string, total_training_positive_cases, total_training_negative_cases, total_eval_positive_cases, total_eval_negative_cases)
    training_data = training_data_arr.reshape((len(training_labels), max_input_height, max_input_width, 1))
    eval_data = eval_data_arr.reshape((len(eval_labels), max_input_height, max_input_width, 1))
    (training_data, training_labels) = shuffle(training_data, training_labels)
    (eval_data, eval_labels) = shuffle(eval_data, eval_labels)
    return (training_data, training_labels, eval_data, eval_labels, max_input_height, max_input_width)

def _retrieve_data_2d(path, max_input_width, max_input_height):
    input = []
    cur_index = 1
    for file in os.listdir(path):
        if file.startswith('.'):
            continue
        with open(os.path.join(path, file), 'r', errors='ignore') as f:
            cur_input = np.zeros((max_input_height, max_input_width), dtype=np.int32)
            is_valid = True
            no_of_lines = 0
            for line in f:
                input_str = line.strip('\n').replace('\t', ' ')
                if input_str == '\n' or input_str == '':
                    if is_valid and max_input_height > no_of_lines > 0:
                        input.append(np.array(cur_input, dtype=np.int32))
                    no_of_lines = 0
                    is_valid = True
                    cur_input = np.zeros((max_input_height, max_input_width), dtype=np.int32)
                    cur_index += 1
                    continue
                if is_valid:
                    arr = np.fromstring(input_str, dtype=np.int32, sep=' ', count=max_input_width)
                    arr_size = len(np.fromstring(input_str, dtype=np.int32, sep=' '))
                    if arr_size > max_input_width:
                        arr_size = max_input_width
                        arr = arr[:arr_size]
                    arr[arr_size:max_input_width] = 0
                    if no_of_lines < max_input_height:
                        cur_input[no_of_lines] = arr
                        no_of_lines += 1
                    else:
                        is_valid = False
    return input

def delete_empty_files(path):
    for (root, dirs, files) in os.walk(path):
        for f in files:
            fullname = os.path.join(root, f)
            try:
                if os.path.getsize(fullname) <= 1:
                    print('\tdeleting ' + fullname)
                    os.remove(fullname)
            except:
                continue

def handle_overloaded_methods(path):
    for (root, dirs, files) in os.walk(path):
        for f in files:
            if not (f.endswith('.tok') or f.endswith('.tok.cld')):
                continue
            if f.endswith('.cld'):
                continue
            filepath = os.path.join(root, f)
            outfilepath = os.path.join(root, f + '.cld')
            with open(outfilepath, 'w') as filewriter:
                with open(filepath, errors='ignore') as f:
                    longest_line = ''
                    for line in f:
                        if len(line) < 2:
                            if len(longest_line) > 1:
                                filewriter.write(longest_line)
                            longest_line = ''
                        elif len(line) > len(longest_line):
                            longest_line = line
            os.remove(filepath)

def _get_outlier_threshold(path, z, is_c2v):
    lengths = []
    for (root, dirs, files) in os.walk(path):
        for f in files:
            if f.startswith('.'):
                continue
            filepath = os.path.join(root, f)
            with open(filepath, 'r', errors='ignore') as file:
                for line in file:
                    input_str = line.replace('\t', ' ')
                    if is_c2v:
                        np_arr = np.fromstring(input_str, dtype=np.float, sep=' ')
                    else:
                        np_arr = np.fromstring(input_str, dtype=np.int32, sep=' ')
                    cur_width = len(np_arr)
                    lengths.append(cur_width)
    return compute_max(lengths, z=z)

def get_outlier_threshold(path, z=1, is_c2v=False):
    len1 = _get_outlier_threshold(os.path.join(path, 'Positive'), z, is_c2v)
    len2 = _get_outlier_threshold(os.path.join(path, 'Negative'), z, is_c2v)
    if len1 > len2:
        return len1
    else:
        return len2

def get_outlier_threshold_2d(path, z=1):
    (width1, height1) = _get_outlier_threshold_2d(os.path.join(path, 'Positive'), z)
    (width2, height2) = _get_outlier_threshold_2d(os.path.join(path, 'Negative'), z)
    width = width2
    if width1 > width2:
        width = width1
    height = height2
    if height1 > height2:
        height = height1
    return (width, height)

def _get_outlier_threshold_2d(path, z=2):
    widths = []
    heights = []
    for (root, dirs, files) in os.walk(path):
        for f in files:
            if f.startswith('.'):
                continue
            filepath = os.path.join(root, f)
            longest_line_length = 0
            no_of_lines = 0
            with open(filepath, 'r', errors='ignore') as f:
                for line in f:
                    symbol_length = len(line.split('\t'))
                    if len(line) < 2:
                        widths.append(longest_line_length)
                        if no_of_lines < 2000:
                            heights.append(no_of_lines)
                        longest_line_length = 0
                        no_of_lines = 0
                    elif symbol_length < 1000:
                        no_of_lines += 1
                        input_str = line.replace('\t', ' ')
                        cur_width = len(np.fromstring(input_str, dtype=np.int32, sep=' '))
                        if cur_width > longest_line_length:
                            longest_line_length = cur_width
    return (compute_max(widths, 'width', z=z), compute_max(heights, 'height', z=z))

def compute_max(arr, dim='width', z=2):
    if len(arr) == 0:
        print(f'WARNING: compute_max received empty array for {dim}. Returning default 500.')
        return 500
    mn = np.mean(arr, axis=0)
    sd = np.std(arr, axis=0)
    final_list = [x for x in arr if x <= mn + z * sd]
    rmn2 = len(arr) - len(final_list)
    print('{} array size '.format(dim) + str(len(arr)))
    print('min {} '.format(dim) + str(min(arr)))
    print('max {} '.format(dim) + str(max(arr)))
    print('mean {} '.format(dim) + str(np.nanmean(arr)))
    print('standard deviation ' + str(np.std(arr)))
    print('median {} '.format(dim) + str(np.nanmedian(arr)))
    print('number of upper outliers removed ' + str(rmn2))
    print('max {} excluding upper outliers '.format(dim) + str(max(final_list)))
    return max(final_list)

def remove_duplicates_1d(path):
    for (root, dirs, files) in os.walk(path):
        for f in files:
            if f.startswith('.'):
                continue
            if f.endswith('.cld'):
                continue
            filepath = os.path.join(root, f)
            outfilepath = os.path.join(root, 'temp.cld')
            if os.path.exists(outfilepath):
                os.remove(outfilepath)
            mydict = dict()
            with open(filepath, errors='ignore') as f:
                for line in f:
                    if line not in mydict:
                        mydict[line] = line
            with open(outfilepath, 'w') as filewriter:
                for value in mydict.values():
                    filewriter.write(value)
            os.remove(filepath)
            os.rename(outfilepath, filepath + '.cld')

def remove_duplicates_c2v(path):
    mydict = dict()
    for case in ['Negative']:
        cur_dir_path = os.path.join(path, case)
        for file in os.listdir(cur_dir_path):
            if file.startswith('.'):
                continue
            if file.endswith('.cvec'):
                continue
            filepath = os.path.join(cur_dir_path, file)
            with open(filepath, errors='ignore') as f:
                for line in f:
                    if line not in mydict:
                        mydict[line] = line
                    else:
                        print('Duplicate found')
        outfilepath = get_out_file(cur_dir_path)
        i = 0
        for value in mydict.values():
            with open(outfilepath, 'a') as filewriter:
                filewriter.write(value)
                i += 1
            if i > 1000:
                i = 0
                if is_file_oversize(outfilepath):
                    outfilepath = get_out_file(cur_dir_path)

def get_out_file(path):
    i = 1
    while True:
        filepath = os.path.join(path, 'samples' + str(i) + '.cvec')
        if not os.path.exists(filepath):
            return filepath
        i = i + 1

def is_file_oversize(filename):
    if os.path.getsize(filename) > 52428800:
        return True
    return False

def remove_duplicates_2d(path):
    for (root, dirs, files) in os.walk(path):
        for f in files:
            if f.startswith('.'):
                continue
            if f.endswith('.cld'):
                continue
            filepath = os.path.join(root, f)
            outfilepath = os.path.join(root, 'temp.cld')
            if os.path.exists(outfilepath):
                os.remove(outfilepath)
            mydict = dict()
            with open(filepath, errors='ignore') as f:
                cur_sample = ''
                for line in f:
                    if len(line) > 1:
                        cur_sample += line
                    else:
                        if not cur_sample in mydict:
                            mydict[cur_sample] = cur_sample
                        cur_sample = ''
            with open(outfilepath, 'w') as filewriter:
                for value in mydict.values():
                    filewriter.write(value + '\n')
            os.remove(filepath)
            os.rename(outfilepath, filepath + '.cld')

def preprocess_data_c2v(tokenizer_out_path):
    print('Preprocessing input data...')
    print('\tRemoving duplicates...')
    remove_duplicates_c2v(tokenizer_out_path)
    print('Preprocessing done.')

def preprocess_data(tokenizer_out_path, dimension=1):
    print('Preprocessing input data...')
    if dimension == 1:
        print('\tRemoving duplicates...')
        remove_duplicates_1d(tokenizer_out_path)
    if dimension == 2:
        print('\tRemoving duplicates...')
        remove_duplicates_2d(tokenizer_out_path)
    print('\tDeleting empty files...')
    delete_empty_files(tokenizer_out_path)
    print('Preprocessing done.')

def preprocess_data_2d(tokenizer_out_path):
    preprocess_data(tokenizer_out_path, 2)

def verify(arr_in, filepath):
    input_str = open(filepath, 'r', errors='ignore').read()
    input_str = input_str.replace('\t', ' ')
    arr = np.fromstring(input_str, dtype=np.int32, sep=' ')
    arr_in = arr_in[0:len(arr)]
    assert np.array_equal(arr, arr_in)

def write_input_data_summary(out_folder, case_str, total_training_positive_cases, total_training_negative_cases, total_eval_positive_cases, total_eval_negative_cases):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    f = open(os.path.join(out_folder, case_str + '.txt'), 'w')
    f.write('tr_pos: ' + str(total_training_positive_cases) + '\n' + 'tr_neg: ' + str(total_training_negative_cases) + '\n' + 'ev_pos: ' + str(total_eval_positive_cases) + '\n' + 'ev_neg: ' + str(total_eval_negative_cases))
    f.close()

def get_predicted_y(prob, threshold):
    out_arr = np.empty(len(prob), dtype=np.float32)
    for i in range(0, len(prob)):
        if prob[i] > threshold:
            out_arr[i] = 1
        else:
            out_arr[i] = 0
    return out_arr

def invert_labels(eval_labels):
    new_labels = np.empty(shape=[len(eval_labels)], dtype=np.float32)
    for i in range(len(eval_labels)):
        if eval_labels[i] == 0:
            new_labels[i] = 1
        else:
            new_labels[i] = 0
    return new_labels