#!/usr/local/bin/python3

import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd


# This function checks XML file for types 4-8 and skips if present
def check_xml_error(input_xml):
    # Make element tree for object
    tree = ET.parse(input_xml)

    # Getting the root of the tree
    root = tree.getroot()

    # Pulling out the name of the image
    image_name_string = root[0][0].text

    # Assigning types other than fluorescent and nonfluor in order to
    # Exit program if list is present
    try:
        root_4 = root[1][4]
        count_4 = len(list(root_4))
    except IndexError:
        print('No root 4 in this file')
        count_4 = 0

    try:
        root_5 = root[1][5]
        count_5 = len(list(root_5))
    except IndexError:
        print('No root 5 in this file')
        count_5 = 0

    try:
        root_6 = root[1][6]
        count_6 = len(list(root_6))
    except IndexError:
        print('No root 6 in this file')
        count_6 = 0

    try:
        root_7 = root[1][7]
        count_7 = len(list(root_7))
    except IndexError:
        print('No root 7 in this file')
        count_7 = 0

    try:
        root_8 = root[1][8]
        count_8 = len(list(root_8))
    except IndexError:
        print('No root 8 in this file')
        count_8 = 0

    # Checking if anything exists in other types
    if (count_4 > 1) or (count_5 > 1) or (count_6 > 1):
        print(f'ERROR: {image_name_string} skipped...contains unknown type.')
        result = 'True'
    else:
        result = 'False'
    # If result = 'True then skipped in main()
    return result, tree, count_7, count_8

# Function that gets X, Y coord for each kernel and labels as fluor or nonfluor
# Dataframe is outputted with this info
# Overall ear stats calculated at end to be shown on pval_plots later
def parse_xml(input_xml, tree, count_7, count_8):
    # Getting the root of the tree
    root = tree.getroot()

    # Pulling out the name of the image
    image_name_string = root[0][0].text

    # Pulling out the fluorescent and non-fluorescent children
    fluorescent = root[1][1]
    nonfluorescent = root[1][2]

    # Setting up some empty lists to move the coordinates from the xml into
    fluor_x = []
    fluor_y = []
    nonfluor_x = []
    nonfluor_y = []

    # If something is listed in root 7...
    if count_7 > 1:
        purple = root[1][7]
        # Getting the coordinates of the purple kernels
        for child in purple:
            if child.tag == 'Marker':
                fluor_x.append(child.find('MarkerX').text)
                fluor_y.append(child.find('MarkerY').text)
    # If something is listed in root 8...
    if count_8 > 1:
        yellow = root[1][8]
        # Getting the coordinates of the yellow kernels
        for child in yellow:
            if child.tag == 'Marker':
                nonfluor_x.append(child.find('MarkerX').text)
                nonfluor_y.append(child.find('MarkerY').text)

    # Getting the coordinates of the fluorescent kernels
    for child in fluorescent:
        if child.tag == 'Marker':
            fluor_x.append(child.find('MarkerX').text)
            fluor_y.append(child.find('MarkerY').text)

    # Getting the coordinates of the non-fluorescent kernels
    for child in nonfluorescent:
        if child.tag == 'Marker':
            nonfluor_x.append(child.find('MarkerX').text)
            nonfluor_y.append(child.find('MarkerY').text)

    # Creating the repeating 'type' column values
    fluor_type = 'Fluorescent'
    nonfluor_type = 'Non-Fluorescent'

    # Putting together the results for output in [file name, type, x coord, y coord] format
    fluor_coord = np.column_stack(([image_name_string] * len(fluor_x), [fluor_type] * len(fluor_x), fluor_x, fluor_y))
    nonfluor_coord = np.column_stack(
        ([image_name_string] * len(nonfluor_x), [nonfluor_type] * len(nonfluor_x), nonfluor_x, nonfluor_y))

    # Stacking the fluor and nonfluor arrays on top of eachother
    combined_array = np.vstack((fluor_coord, nonfluor_coord))

    # Importing np.array into pandas dataframe and converting coordinate values from objects to integers
    df = pd.DataFrame(data=combined_array, columns='File Type X-Coordinate Y-Coordinate'.split())
    df['X-Coordinate'] = df['X-Coordinate'].astype(np.int64)
    df['Y-Coordinate'] = df['Y-Coordinate'].astype(np.int64)

    df.to_csv('xml_coord.txt', sep='\t')
    return df

thing, tree, count_7, count_8 = check_xml_error("/Users/elysevischulis/PycharmProjects/Maize_Scanner/Data for Analysis/Single XML/X401x492-2m1.xml")
thing2 = parse_xml(thing, tree, count_7, count_8)