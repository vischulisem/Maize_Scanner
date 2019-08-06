#!/usr/local/bin/python3

# This script contains functions to extract XML kernel coordinates and put them into a dataframe
# Labelling them as fluor or nonfluor, stats on entire ear are collected
# XML files containing types 4-8 are skipped
# Next, Sliding_Window function generates a new dataframe based on desired width and steps of window
# Ear is scanned based on window and total kernels, fluor, and nonfluor are calculated for each window
# Finally, a plot is generated comparing percent transmission across different windows on the ear
# Chi squared test is calculated for each window
# Line plotted is colored based on whether points are above or below p = 0.05
# Regression line is added to plot, plots saved to new directory with xml file as name of each plot

# Input arguments are an XML file or directory, width, and steps
# Output is positional transmission plots saved to a folder containing 'XML_filename'.png
# Meta dataframe outputted and saved as .txt containing overall ear stats
# Everything dataframe outputted and saved as .txt containing window calculations including percent transmission
# per window and P-values for each window

import sys
import os
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import pylab as pl
from matplotlib import collections as mc
import seaborn as sns


# Setting up argparse arguments
parser = argparse.ArgumentParser(description='Given XML file, width, and steps, returns scatterplot')
parser.add_argument('-x', '--xml', metavar='', help='Input XML filename.', type=str)
parser.add_argument('-w', '--width', metavar='', help='Width in pixels for the length of the window.', default=400, type=int)
parser.add_argument('-s', '--step_size', metavar='', help='Steps in pixels for window movement.', default=2, type=int)
parser.add_argument('-n', action='store_true', help='Will normalize x axis of transmission plots.')
parser.add_argument('-tk', '--total_kernels', metavar='', help='Determine threshold for total kernels on ear for skipping file.', default=50, type=int)
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

    # Overall ear stats
    # Counting total number of kernels per ear
    overall_kernel_total = int(len(df.index))
    # Calculating expected value for chi squared test
    overall_expected = overall_kernel_total * 0.5

    # Counting number of fluorescent
    if any(df.Type == 'Fluorescent'):
        x = df['Type'].value_counts()
        overall_fluor_tot = x['Fluorescent']
    else:
        overall_fluor_tot = 0

    # Calculating percent transmission for entire ear
    overall_perc_trans = overall_fluor_tot/overall_kernel_total

    # Counting number nonfluorescent
    if any(df.Type == 'Non-Fluorescent'):
        x = df['Type'].value_counts()
        overall_nonfluor_tot = x['Non-Fluorescent']
    else:
        overall_nonfluor_tot = 0

    # Chi squared test for entire ear..stats returned to be used in pval_plot and displayed in txt box
    chi_stat = stats.chisquare([overall_fluor_tot, overall_nonfluor_tot], [overall_expected, overall_expected])
    overall_pval = chi_stat[1]

    return df, overall_kernel_total, overall_perc_trans, overall_pval, overall_fluor_tot, overall_nonfluor_tot

# Creates sliding parameter to count total kernels, fluor, and nonfluor on ear as you move across
# left to right. User inputs the desired window length and step size in pixels.
def sliding_window(df, w, s, filename):
    # sort x values from small to big
    df.sort_values(by=['X-Coordinate'], inplace=True)
    df = df.reset_index(drop=True)

    # Choosing starting point for window with value for x
    start_x = df["X-Coordinate"].head(1)
    # Converting to int
    int_start_x = int(start_x)

    # Setting up w and s as integers
    int_w = int(w)
    int_s = int(s)

    # Defining end of window
    end_x = int_start_x + int_w

    # Defining steps for window
    steps = int_s

    # Creating empty dataframe for window calculations to be appended to in loop later
    kern_count_df = pd.DataFrame(
        columns='File Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor'.split())

    # Assigning variable to final x coordinate in dataframe
    final_x_coord = df["X-Coordinate"].tail(1)
    # and converting to int
    int_fxc = int(final_x_coord)

    # Creating error messages for too small or too big input width or steps
    adj_step_fxc = int_fxc * 0.25

    if end_x >= int_fxc:
        print(f'For {filename} - Width of window is too large, enter smaller width value.')
        ans1 = 'True'
    else:
        ans1 = 'False'

    if int_s >= adj_step_fxc:
        print(f'For {filename} - Step size is too large, enter smaller value.')
        ans2 = 'True'
    else:
        ans2 = 'False'

    # Beginning of sliding window to scan ear and output new dataframe called kern_count_df
    while end_x <= int_fxc:

        # Creating smaller window df based on original df this will slide across ear
        rslt_df = df[(df['X-Coordinate'] >= int_start_x) & (df['X-Coordinate'] <= end_x)]

        # Total kernels in window
        kernel_tot = len(rslt_df.index)

        # Error message if there are no kernels in window
        if kernel_tot == 0:
            print(f'For {filename} - 0 Kernels in Window, please enter larger width value.')
            ans3 = 'True'
            break
        else:
            ans3 = 'False'

        # Listing start and end pixels for window
        window_start = int_start_x
        window_end = end_x

        # Counting the number of fluor and nonfluor in window
        if any(rslt_df.Type == 'Fluorescent'):
            x = rslt_df['Type'].value_counts()
            fluor_tot = x['Fluorescent']
        else:
            fluor_tot = 0

        if any(rslt_df.Type == 'Non-Fluorescent'):
            x = rslt_df['Type'].value_counts()
            nonfluor_tot = x['Non-Fluorescent']
        else:
            nonfluor_tot = 0

        # Creating list with variables we just calculated
        data = [[filename, steps, window_start, window_end, kernel_tot, fluor_tot, nonfluor_tot]]

        # Putting list into dataframe (which is just 1 row)
        data_df = pd.DataFrame(data=data,
                               columns='File Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor'.split())

        # Appending data_df to kern_count_df (1 row added each time)
        kern_count_df = kern_count_df.append(data_df)

        # Shifting window based on step size
        int_start_x = int_start_x + steps
        end_x = end_x + steps

    # Resetting index
    kern_count_df = kern_count_df.reset_index(drop=True)
    # Converting numbers to ints
    cols = kern_count_df.columns.drop(['File'])
    kern_count_df[cols] = kern_count_df[cols].apply(pd.to_numeric)

    # Also part of error messages
    ans3 = 'Neither'

    return kern_count_df, ans1, ans2, ans3

# Computes chisquare test on fluor and nonfluor for each window compared to
# total kernels per window. Dataframe with p-values outputted.
def chisquare_test(kern_count_df):
    # Setting up index to start at 0
    index = 0
    # Deciding where to stop loop (at the last index)
    end_index = kern_count_df.index[-1]
    # Creating empty df to append each row from loop
    temp_df = pd.DataFrame(columns='P-Value Comparison'.split())

    # Beginning of loop to calculate chi squared for each window
    while index <= end_index:
        # Narrowing down df to single row with only rows for calculations
        single_row = kern_count_df.iloc[[index]]
        single_row = single_row.loc[:, 'Total_Kernels':'Total_NonFluor']

        # Assigning expected and variables for chisquare
        expected = single_row['Total_Kernels'].values[0] * 0.5
        fluor = single_row['Total_Fluor'].values[0]
        nonfluor = single_row['Total_NonFluor'].values[0]

        # Calculating chi square
        chi_stat = stats.chisquare([fluor, nonfluor], [expected, expected])
        # Setting variable for p-value
        pval = chi_stat[1]

        # Creating new column with labels based on p-value
        if pval <= 0.05:
            p_input = '< p = 0.05'
        else:
            p_input = '> p = 0.05'

        # Putting variables into list
        data = [[pval, p_input]]
        # Putting list into dataframe (single row)
        data_df = pd.DataFrame(data=data, columns='P-Value Comparison'.split())
        # Appending to empty df outside of loop
        temp_df = temp_df.append(data_df)
        # Repeats for next row
        index = index + 1

    # Resetting index and sticking df with p-values to existing kern_count df
    temp_df = temp_df.reset_index(drop=True)
    final_df = pd.concat([kern_count_df, temp_df], axis=1, sort=False)
    final_df = final_df.reset_index(drop=True)

    return final_df

# Plots a line for percent transmission across the ear
# colored based on whether point is above or below p = 0.05
# Does normalize window mean x axis
def pval_plot(final_df, xml, overall_kernel_total, overall_perc_trans, overall_pval, count_7, path):
    # Hiding error message
    plt.rcParams.update({'figure.max_open_warning': 0})

    # Creating new column of 'window mean'
    col = final_df.loc[:, "Window_Start":"Window_End"]
    final_df['window_mean'] = col.mean(axis=1)
    # Normalizing window mean values for x axis
    final_df['Normalized_Window_Mean'] = final_df.groupby('File')['window_mean'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Creating new column of percent transmission for each window
    final_df['Percent_Transmission'] = final_df['Total_Fluor'] / final_df['Total_Kernels']

    # Counting number of rows in dataframe
    end_index = final_df.index[-1]

    # Calculating regression
    reg_x = final_df['Normalized_Window_Mean'].values
    reg_y = final_df['Percent_Transmission'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
    rsq = r_value ** 2

    # Beginning of plotting... using line collections to plot segments that are
    # colored based on tuples (red or blue) for p values
    segments = []
    colors = np.zeros(shape=(end_index, 4))
    x = final_df['Normalized_Window_Mean'].values
    y = final_df['Percent_Transmission'].values
    z = final_df['P-Value'].values
    i = 0

    for x1, x2, y1, y2, z1, z2 in zip(x, x[1:], y, y[1:], z, z[1:]):
        if z1 > 0.05:
            colors[i] = tuple([1, 0, 0, 1])
        elif z1 <= 0.05:
            colors[i] = tuple([0, 0, 1, 1])
        else:
            colors[i] = tuple([0, 1, 0, 1])
        segments.append([(x1, y1), (x2, y2)])
        i += 1

    # Creating a line by putting segments together
    lc = mc.LineCollection(segments, colors=colors, linewidths=2)
    fig, ax = pl.subplots(figsize=(11.7, 8.27))
    # Adding line collection to axis
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)

    # Plotting regression line
    plt.plot(reg_x, intercept + slope * reg_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])

    # Settings for graph aesthetics
    # ax.set_xlim(np.min(x)-50, np.max(x)+50)
    ax.set_ylim(0, 1)
    plt.yticks(np.arange(0, 1, step=0.25))
    plt.figure(figsize=(11.7, 8.27))

    ax.set_title(xml[:-4]+' Plot', fontsize=30, fontweight='bold')
    ax.set_xlabel('Normalized Window Position (pixels)', fontsize=20, fontweight='bold')
    ax.set_ylabel('% GFP', fontsize=20, fontweight='bold')

    ax.set_facecolor('white')
    ax.yaxis.grid(color='grey')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')

    # Key to label line colors
    red_patch = mpatches.Patch(color='red', label='> p = 0.05')
    blue_patch = mpatches.Patch(color='blue', label='< p = 0.05')
    ax.legend(handles=[red_patch, blue_patch], loc='center left', bbox_to_anchor=(1, 0.5))

    # Creating a text box with overall stats for each graph
    num_weird_trans = len(final_df[final_df['Comparison'] == '< p = 0.05'])
    num_tkern = int(len(final_df))

    window_stat = num_weird_trans / num_tkern
    window_stat = round(window_stat, 3)

    overall_kernel_total = round(overall_kernel_total, 3)
    overall_perc_trans = round(overall_perc_trans, 3)
    overall_pval = '{:0.3e}'.format(overall_pval)
    slope = round(slope, 5)
    intercept = round(intercept, 3)
    rsq = round(rsq, 3)
    p_value = '{:0.3e}'.format(p_value)

    if count_7 > 1:
        purple = 'Yes'
    else:
        purple = 'No'

    textstr = '\n'.join((f'Overall Total Kernels = {overall_kernel_total}',
                         f'Overall Percent Transmission = {overall_perc_trans}',
                         f'Overall ChiSquared P-Value = {overall_pval}',
                         f'Purple Kernels Present? {purple}',
                         f'% Windows not 0.5 Transmission = {window_stat}',
                         f'Regression Slope = {slope}',
                         f'Regression Intercept = {intercept}',
                         f'Regression R-squared = {rsq}',
                         f'Regression P-Value = {p_value}'))

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, fontweight='bold', verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10, 'edgecolor': 'black'})

    fig.canvas.draw()

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[6] = 'Top'
    labels[1] = 'Bottom'
    labels[2] = 0.2
    labels[3] = 0.4
    labels[4] = 0.6
    labels[5] = 0.8

    ax.set_xticklabels(labels)
    # Getting the figure for saving etc
    pv_plot = lc.get_figure()

    # Create directory to save plots
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Male_plot_folder/Transmission_Norm_plots/')
    # Sample_file_name
    sample_file_name = xml[:-4] + '.png'

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    pv_plot.savefig(results_dir + sample_file_name, bbox_inches="tight")
    plt.close()

# Creating temporary dataframe that will append to the meta df in main

    total_kernels = len(final_df.index)
    total_fluor = final_df['Total_Fluor'].sum()
    perc_trans = total_fluor/total_kernels

    all_data = [[xml[:-4], total_kernels, perc_trans, r_value ** 2, p_value, slope]]

    # Putting list into dataframe (which is just 1 row)
    data_df = pd.DataFrame(data=all_data, columns='File_Name Total_Kernels Percent_Transmission R-Squared P-Value Slope'.split())

    return pv_plot, data_df

# Plots a line for percent transmission across the ear
# colored based on whether point is above or below p = 0.05
# Doesn't normalize window mean x axis
def pval_notnorm_plot(final_df, xml, overall_kernel_total, overall_perc_trans, overall_pval, count_7, path):
    # Hiding error message
    plt.rcParams.update({'figure.max_open_warning': 0})

    # Creating new column of 'window mean'
    col = final_df.loc[:, "Window_Start":"Window_End"]
    final_df['window_mean'] = col.mean(axis=1)
    final_df['Normalized_Window_Mean'] = final_df.groupby('File')['window_mean'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Creating new column of percent transmission for each window
    final_df['Percent_Transmission'] = final_df['Total_Fluor'] / final_df['Total_Kernels']

    # Counting number of rows in dataframe
    end_index = final_df.index[-1]

    # Calculating regression
    reg_x = final_df['window_mean'].values
    reg_y = final_df['Percent_Transmission'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
    rsq = r_value ** 2

    # Beginning of plotting... using line collections to plot segments that are
    # colored based on tuples (red or blue) for p values
    segments = []
    colors = np.zeros(shape=(end_index, 4))
    x = final_df['window_mean'].values
    y = final_df['Percent_Transmission'].values
    z = final_df['P-Value'].values
    i = 0

    for x1, x2, y1, y2, z1, z2 in zip(x, x[1:], y, y[1:], z, z[1:]):
        if z1 > 0.05:
            colors[i] = tuple([1, 0, 0, 1])
        elif z1 <= 0.05:
            colors[i] = tuple([0, 0, 1, 1])
        else:
            colors[i] = tuple([0, 1, 0, 1])
        segments.append([(x1, y1), (x2, y2)])
        i += 1

    # Creating a line by putting segments together
    lc = mc.LineCollection(segments, colors=colors, linewidths=2)
    fig, ax = pl.subplots(figsize=(11.7, 8.27))
    # Adding line collection to axis
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)

    # Plotting regression line
    plt.plot(reg_x, intercept + slope * reg_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])

    # Settings for graph aesthetics
    ax.set_xlim(np.min(x)-50, np.max(x)+50)
    ax.set_ylim(0, 1)
    plt.yticks(np.arange(0, 1, step=0.25))
    plt.figure(figsize=(11.7, 8.27))

    ax.set_title(xml[:-4]+' Plot', fontsize=30, fontweight='bold')
    ax.set_xlabel('Window Position (pixels)', fontsize=20, fontweight='bold')
    ax.set_ylabel('% GFP', fontsize=20, fontweight='bold')

    ax.set_facecolor('white')
    ax.yaxis.grid(color='grey')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')

    # Key to label line colors
    red_patch = mpatches.Patch(color='red', label='> p = 0.05')
    blue_patch = mpatches.Patch(color='blue', label='< p = 0.05')
    ax.legend(handles=[red_patch, blue_patch], loc='center left', bbox_to_anchor=(1, 0.5))

    # Creating a text box with overall stats for each graph
    num_weird_trans = len(final_df[final_df['Comparison'] == '< p = 0.05'])
    num_tkern = int(len(final_df))

    window_stat = num_weird_trans / num_tkern
    window_stat = round(window_stat, 3)

    overall_kernel_total = round(overall_kernel_total, 3)
    overall_perc_trans = round(overall_perc_trans, 3)
    overall_pval = '{:0.3e}'.format(overall_pval)
    slope = round(slope, 5)
    intercept = round(intercept, 3)
    rsq = round(rsq, 3)
    p_value = '{:0.3e}'.format(p_value)

    if count_7 > 1:
        purple = 'Yes'
    else:
        purple = 'No'

    textstr = '\n'.join((f'Overall Total Kernels = {overall_kernel_total}',
                         f'Overall Percent Transmission = {overall_perc_trans}',
                         f'Overall ChiSquared P-Value = {overall_pval}',
                         f'Purple Kernels Present? {purple}',
                         f'% Windows not 0.5 Transmission = {window_stat}',
                         f'Regression Slope = {slope}',
                         f'Regression Intercept = {intercept}',
                         f'Regression R-squared = {rsq}',
                         f'Regression P-Value = {p_value}'))

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, fontweight='bold', verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10, 'edgecolor': 'black'})

    # Getting the figure for saving etc
    pv_plot = lc.get_figure()

    # Create directory to save plots
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Male_plot_folder/Transmission_plots/')
    # Sample_file_name
    sample_file_name = xml[:-4] + '.png'

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    pv_plot.savefig(results_dir + sample_file_name, bbox_inches="tight")
    plt.close()

# Creating temporary dataframe that will append to the meta df in main

    total_kernels = len(final_df.index)
    total_fluor = final_df['Total_Fluor'].sum()
    perc_trans = total_fluor/total_kernels

    all_data = [[xml[:-4], total_kernels, perc_trans, r_value ** 2, p_value, slope]]

    # Putting list into dataframe (which is just 1 row)
    data_df = pd.DataFrame(data=all_data, columns='File_Name Total_Kernels Percent_Transmission R-Squared P-Value Slope'.split())

    return pv_plot, data_df

def male_fam_plot(input_df, path):
    print(f'Starting male plots...')
    # Plotting begins
    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    sns.lineplot(x="Normalized_Window_Mean", y="Percent_Transmission", data=input_df, hue="File", linewidth=5)
    plt.ylim(0, 1)
    ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.title('Male Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
    plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
    plt.ylabel('Percent Transmission', fontsize=18, weight='bold')
    ax.set_facecolor('white')
    ax.yaxis.grid(color='gainsboro')
    # Regression calculations
    reg_x = input_df['Normalized_Window_Mean'].values
    reg_y = input_df['Percent_Transmission'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
    r2 = r_value ** 2

    # Plotting regression line
    plt.plot(reg_x, intercept + slope * reg_x, 'r', label='fitted line', color='red', linewidth=3,
                     dashes=[5, 3])

    # Creating empty df so that mean rsquared for each line can be displayed
    stat_df = pd.DataFrame(columns='Slope Intercept RSquared P-Value'.split())
    # Iterating through each file name and plotting regression line
    grps = input_df.groupby(['File'])
    for file, grp in grps:
        iter_y = grp['Percent_Transmission']
        iter_x = grp['Normalized_Window_Mean']
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(iter_x, iter_y)
        plt.plot(iter_x, intercept2 + slope2 * iter_x, 'r', label='fitted line', color='black', linewidth=3,
                         dashes=[5, 3])
        # Putting regression stats for each file into dataframe
        this_stat = [[slope2, intercept2, r_value2 ** 2, p_value2]]
        temp_stat_df = pd.DataFrame(data=this_stat, columns='Slope Intercept RSquared P-Value'.split())
        stat_df = stat_df.append(temp_stat_df)
        stat_df = stat_df.reset_index(drop=True)

    # Calculating mean values for each stat and rounding to 4 decimal places
    average_slope = round(stat_df['Slope'].mean(), 4)
    average_intercept = round(stat_df['Intercept'].mean(), 4)
    average_Rsquared = round(stat_df['RSquared'].mean(), 4)
    average_Pval = '{:0.3e}'.format(stat_df['P-Value'].mean())
    combined_pval = '{:0.3e}'.format(p_value)
    # Text string for text box
    textstr = '\n'.join((f'Combined Slope = {round(slope, 4)}',
                         f'Average Indv. Slope = {average_slope}',
                         f'Combined Intercept = {round(intercept, 4)}',
                         f'Average Indv. Intercept = {average_intercept}',
                         f'Combined P-value = {combined_pval}',
                         f'Average Indv. P-value = {average_Pval}',
                         f'Combined R-Squared = {round(r_value ** 2, 4)}',
                         f'Average Indv. R-squared = {average_Rsquared}'))
    # Creating text box on graph
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, fontweight='bold',
                    verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10, 'edgecolor': 'black'})

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')

    fig.canvas.draw()

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[6] = 'Top'
    labels[1] = 'Bottom'
    labels[2] = 0.2
    labels[3] = 0.4
    labels[4] = 0.6
    labels[5] = 0.8

    ax.set_xticklabels(labels)

    # Saving figure in new directory with cross file name
    male_graph = plt.gcf()

    # create directory to save plots
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Male_plot_folder/Male_Plots/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # sample_file_name
    sample_file_name = 'maleplot.png'

    male_graph.savefig(results_dir + sample_file_name, bbox_inches="tight")
    plt.close()
    print(f'Done male plots!')

    return plt

# Main function for running the whole script with argparse
# if - allows you to input xml file as first argument
# else - allows you to input directory of xml files as argument
def main():
    # Dataframe (saved to .txt file) for each file and their chisquare and regression stats
    meta_df = pd.DataFrame(columns='File_Name Total_Kernels Percent_Transmission R-Squared P-Value Slope'.split())
    # Dataframe (saved to .txt file) for each file's name, window_mean, % transmission, pvalue, etc
    everything_df = pd.DataFrame(columns='File Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor P-Value Comparison window_mean Normalized_Window_Mean Percent_Transmission'.split())

# Processing single file as argument
    if args.xml.endswith(".xml"):
        result, tree, count_7, count_8 = check_xml_error(args.xml)
        if result == 'True':
            sys.exit('Program Exit')
        # check xml error fun
        print(f'Processing {args.xml}...')
        dataframe, overall_kernel_total, overall_perc_trans, overall_pval, overall_fluor_tot, overall_nonfluor_tot = parse_xml(args.xml, tree, count_7, count_8)
        if overall_kernel_total <= args.total_kernels:
            sys.exit(f'{args.xml} skipped...overall kernel total less than {args.total_kernels}')
        if overall_fluor_tot == 0:
            sys.exit(f'{args.xml} skipped...no fluorescent kernels on ear.')
        if overall_nonfluor_tot == 0:
            sys.exit(f'{args.xml} skipped...no nonfluorescent kernels on ear.')
        dataframe2, ans1, ans2, ans3 = sliding_window(dataframe, args.width, args.step_size, args.xml)
        if (ans1 == 'True') or (ans2 == 'True') or (ans3 == 'True'):
            sys.exit('Program Exit')
        chi_df = chisquare_test(dataframe2)
        if args.n:
            trans_plot, end_df = pval_plot(chi_df, args.xml, overall_kernel_total, overall_perc_trans, overall_pval,
                                           count_7, args.path)
        else:
            trans_plot, end_df = pval_notnorm_plot(chi_df, args.xml, overall_kernel_total, overall_perc_trans,
                                                   overall_pval, count_7, args.path)
        everything_df = everything_df.append(chi_df)
        everything_df = everything_df.reset_index(drop=True)
        meta_df = meta_df.append(end_df)
        meta_df = meta_df.reset_index(drop=True)
# Processing directory of xml files
    else:
        for roots, dirs, files in os.walk(args.xml):
            for filename in files:
                fullpath = os.path.join(args.xml, filename)
                print(f'Processing {fullpath}...')
                if fullpath.endswith(".xml"):
                    with open(fullpath, 'r') as f:
                        result, tree, count_7, count_8 = check_xml_error(f)
                        if result == 'True':
                            continue
                        dataframe, overall_kernel_total, overall_perc_trans, overall_pval, overall_fluor_tot, overall_nonfluor_tot = parse_xml(f, tree, count_7, count_8)
                        if overall_kernel_total <= args.total_kernels:
                            print(f'{filename} skipped...overall kernel total less than {args.total_kernels}')
                            continue
                        if overall_fluor_tot == 0:
                            print(f'{filename} skipped...no fluorescent kernels on ear.')
                            continue
                        if overall_nonfluor_tot == 0:
                            print(f'{filename} skipped...no nonfluorescent kernels on ear.')
                            continue
                        dataframe2, ans1, ans2, ans3 = sliding_window(dataframe, args.width, args.step_size, filename)
                        if (ans1 == 'True') or (ans2 == 'True') or (ans3 == 'True'):
                            continue
                        chi_df = chisquare_test(dataframe2)
                        if args.n:
                            trans_plot, end_df = pval_plot(chi_df, filename, overall_kernel_total, overall_perc_trans, overall_pval, count_7, args.path)
                        else:
                            trans_plot, end_df = pval_notnorm_plot(chi_df, filename, overall_kernel_total, overall_perc_trans,
                                                           overall_pval, count_7, args.path)
                        everything_df = everything_df.append(chi_df)
                        everything_df = everything_df.reset_index(drop=True)
                        meta_df = meta_df.append(end_df)
                        meta_df = meta_df.reset_index(drop=True)

    male_fam_plot(everything_df, args.path)

    print('Process Complete!')

if __name__ == '__main__':
    main()


