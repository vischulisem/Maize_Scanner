#!/usr/local/bin/python3

# This script makes a scatterplot based on the coordinates for each kernel in the xml file
# Points are colored based on fluorescence
# Plots are saved to new directory
# Can process single xml file or directory of xml files

import sys
import os
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up argparse arguments
parser = argparse.ArgumentParser(description='Given XML file(s), scatterplot of fluor and nonfluor kernels.')
parser.add_argument('-x', '--xml', metavar='', help='Input XML filename or directory.', type=str)
parser.add_argument('-p', '--path', metavar='', help='List path where you want files saved to.', default=os.getcwd(), type=str)
args = parser.parse_args()

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

    try:
        root_8 = root[1][8]
    except IndexError:
        print('No root 8 in this file')

    # Checking if anything exists in other types
    if (count_4 > 1) or (count_5 > 1) or (count_6 > 1):
        print(f'ERROR: {image_name_string} skipped...contains unknown type.')
        result = 'True'
    else:
        result = 'False'
    # If result = 'True then skipped in main()
    return result, tree

# Function that gets X, Y coord for each kernel and labels as fluor or nonfluor
# Dataframe is outputted with this info
# Overall ear stats calculated at end to be shown on pval_plots later
def parse_xml(input_xml, tree):
    # Getting the root of the tree
    root = tree.getroot()

    # Pulling out the name of the image
    image_name_string = root[0][0].text

    # Pulling out the fluorescent and non-fluorescent children
    fluorescent = root[1][1]
    nonfluorescent = root[1][2]
    purple = root[1][7]
    yellow = root[1][8]

    # Setting up some empty lists to move the coordinates from the xml into
    fluor_x = []
    fluor_y = []
    nonfluor_x = []
    nonfluor_y = []

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

    # Getting the coordinates of the purple kernels
    for child in purple:
        if child.tag == 'Marker':
            fluor_x.append(child.find('MarkerX').text)
            fluor_y.append(child.find('MarkerY').text)

    # Getting the coordinates of the yellow kernels
    for child in yellow:
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

    return df, image_name_string

# Generating plot of coordinate values on ear of fluor and nonfluor
def make_scatter(df, xml, path):
    # Setting background as white
    sns.set_style("white")
    # Setting figure size
    sns.set(rc={'figure.figsize': (9, 2.5)})
    # Creating seaborn scatterplot called 'ax'
    ax = sns.scatterplot("X-Coordinate", "Y-Coordinate", hue="Type", data=df, palette='Set1')
    # Creating Legend
    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # Telling legend where to be placed
    ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    plt.axis('equal')
    figure = ax.get_figure()

    # Create directory to save plots
    script_dir = path
    results_dir = os.path.join(script_dir, 'Kern_Coord_plots/')
    # Sample_file_name
    sample_file_name = xml[:-4] + '.png'

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # Saving figure
    figure.savefig(results_dir + sample_file_name, bbox_inches="tight")
    plt.close()

    return figure

# Main function for running the whole script with argparse
# if - allows you to input xml file as first argument
# else - allows you to input directory of xml files as argument
def main():
    # Processing single file as argument
    if args.xml.endswith(".xml"):
        result, tree = check_xml_error(args.xml)
        if result == 'True':
            sys.exit('Program Exit')
        # check xml error fun
        print(f'Processing {args.xml}...')
        dataframe, image_name = parse_xml(args.xml, tree)
        make_scatter(dataframe, image_name, args.path)
    # Processing directory of xml files
    else:
        for roots, dirs, files in os.walk(args.xml):
            for filename in files:
                fullpath = os.path.join(args.xml, filename)
                print(f'Processing {fullpath}...')
                if fullpath.endswith(".xml"):
                    with open(fullpath, 'r') as f:
                        result, tree = check_xml_error(f)
                        if result == 'True':
                            continue
                        dataframe, image_name = parse_xml(f, tree)
                        make_scatter(dataframe, image_name, args.path)

if __name__ == '__main__':
    main()