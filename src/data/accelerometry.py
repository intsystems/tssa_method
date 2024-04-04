""" complementary module with data extraction methods from raw data files
"""
import pandas as pd
import numpy as np


def get_accelerometry_data(type_label: int, participant_number: int):
    """ return acceleration vector time series for chosen participant and activity type

    :return np.ndarray: array (time, acceler_components)
    """
    # descriptors of file's with accelerometry
    acc_x_file = open(r'../data/UCI_HAR_Dataset/train/Inertial Signals/body_acc_x_train.txt', 'r')
    acc_y_file = open(r'../data/UCI_HAR_Dataset/train/Inertial Signals/body_acc_y_train.txt', 'r')
    acc_z_file = open(r'../data/UCI_HAR_Dataset/train/Inertial Signals/body_acc_z_train.txt', 'r')
    data_files = [acc_x_file, acc_y_file, acc_z_file]

    # with labels representing types of activity of participants
    labels_file = open(r'../data/UCI_HAR_Dataset/train/y_train.txt', 'r')

    # with particapants identifiers
    participants_file = open(r'../data/UCI_HAR_Dataset/train/subject_train.txt', 'r')

    # container for time series
    data = []

    # change parameters to strings
    type_label = str(type_label)
    participant_number = str(participant_number)
    # row numbers boundaries in train data with interested time series
    row_line_first = 0
    row_line_last = 0

    # finding first row line of interest
    while True:
        cur_label = labels_file.readline()
        cur_participant = participants_file.readline()

        if cur_label[0] == type_label and cur_participant[0] == participant_number:
            row_line_last = row_line_first + 1

            # now find last row line of interest
            while True:
                cur_label = labels_file.readline()
                cur_participant = participants_file.readline()

                if cur_label[0] != type_label or cur_participant[0] != participant_number:
                    break

                row_line_last += 1   

            break

        row_line_first += 1


    for data_file in data_files:
        # extracting accelerometry
        current_acc = data_file.readlines()[row_line_first: row_line_last]
        current_acc = ''.join(current_acc)
        current_acc = current_acc.split(' ')
        # removing '\n'
        current_acc = list(filter(lambda x: x != '\n', current_acc))
        # removing '' elements
        current_acc = list(filter(lambda x: x != '', current_acc))
        # casting to float values
        current_acc = list(map(lambda x: float(x), current_acc))
        data.append(current_acc)

    data = np.array(data).T

    return data
