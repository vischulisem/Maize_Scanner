#!/usr/local/bin/python3

# This script is a mirror of XML_to_ChiSquareTrasmPlot.py except it creates 5 models
# that randomly generate whether a kernel is fluorescent or nonfluorescent.
# It then graphs each model in a similar fashion, the line being colored whether it is above
# or below p = 0.05 from the chi square test.
# Regression line is also calculated and plotted.
# Meta_model_df.txt and everything_model_df.txt are outputted with new model calculations and values.

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

# Setting up argparse arguments
parser = argparse.ArgumentParser(description='Given XML file, width, and steps, returns scatterplot')
parser.add_argument('-x', '--xml', metavar='', help='Input XML filename.', type=str)
parser.add_argument('-w', '--width', metavar='', help='Width in pixels for the length of the window.', default=400, type=int)
parser.add_argument('-s', '--step_size', metavar='', help='Steps in pixels for window movement.', default=2, type=int)
parser.add_argument('-n', action='store_true', help='Will normalize x axis of transmission plots.')
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
	if (count_4 > 1) or (count_5 > 1) or (count_6 > 1) or (count_7 > 1) or (count_8 > 1):
		print(f'ERROR: {image_name_string} skipped...contains unknown type.')
		result = 'True'
	else:
		result = 'False'

	return result, tree

# Function that gets X, Y coord for each kernel and labels as fluor or nonfluor
# Dataframe is outputted with this info
# Models generated
# Overall ear stats calculated at end to be shown on pval_plots later
def parse_xml(input_xml, tree):
	# Make element tree for object
	# tree = ET.parse(input_xml)

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

	# Randomly assigning fluor or nonfluor to each coordinate
	# 1 = fluor, 2 = nonfluor
	df['randGFP1'] = np.random.randint(1, 3, df.shape[0])
	df['randGFP2'] = np.random.randint(1, 3, df.shape[0])
	df['randGFP3'] = np.random.randint(1, 3, df.shape[0])
	df['randGFP4'] = np.random.randint(1, 3, df.shape[0])
	df['randGFP5'] = np.random.randint(1, 3, df.shape[0])

	# Converting values to ints
	df['X-Coordinate'] = df['X-Coordinate'].astype(np.int64)
	df['Y-Coordinate'] = df['Y-Coordinate'].astype(np.int64)
	df = df.reset_index(drop=True)
	return df

# Creates sliding parameter to count total kernels, fluor, and nonfluor on ear as you move across
# left to right. User inputs the desired window length and step size in pixels.
def sliding_window(df, w, s, filename):
	# Sort x values from small to big
	df.sort_values(by=['X-Coordinate'], inplace=True)
	df = df.reset_index(drop=True)

	# Choosing starting point for window with value for x
	start_x = df["X-Coordinate"].head(1)
	int_start_x = int(start_x)

	# Setting up w and s as integers
	int_w = int(w)
	int_s = int(s)

	# Defining end of window
	end_x = int_start_x + int_w

	# Defining steps for window
	steps = int_s

	# Creating empty dataframe
	kern_count_df = pd.DataFrame(
		columns='File Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor Fluor2 Nonfluor2 Fluor3 Nonfluor3 Fluor4 Nonfluor4 Fluor5 Nonfluor5'.split())

	# Assigning variable to final x coordinate in dataframe
	final_x_coord = df["X-Coordinate"].tail(1)
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

		# Creating smaller window df based on original df
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
		if any(rslt_df.randGFP1 == 1):
			x = rslt_df['randGFP1'].value_counts()
			fluor_tot = x[1]
		else:
			fluor_tot = 0

		if any(rslt_df.randGFP1 == 2):
			x = rslt_df['randGFP1'].value_counts()
			nonfluor_tot = x[2]
		else:
			nonfluor_tot = 0

		if any(rslt_df.randGFP2 == 1):
			x = rslt_df['randGFP2'].value_counts()
			fluor2_tot = x[1]
		else:
			fluor2_tot = 0

		if any(rslt_df.randGFP2 == 2):
			x = rslt_df['randGFP2'].value_counts()
			nonfluor2_tot = x[2]
		else:
			nonfluor2_tot = 0

		if any(rslt_df.randGFP3 == 1):
			x = rslt_df['randGFP3'].value_counts()
			fluor3_tot = x[1]
		else:
			fluor3_tot = 0

		if any(rslt_df.randGFP3 == 2):
			x = rslt_df['randGFP3'].value_counts()
			nonfluor3_tot = x[2]
		else:
			nonfluor3_tot = 0

		if any(rslt_df.randGFP4 == 1):
			x = rslt_df['randGFP4'].value_counts()
			fluor4_tot = x[1]
		else:
			fluor4_tot = 0

		if any(rslt_df.randGFP4 == 2):
			x = rslt_df['randGFP4'].value_counts()
			nonfluor4_tot = x[2]
		else:
			nonfluor4_tot = 0

		if any(rslt_df.randGFP5 == 1):
			x = rslt_df['randGFP5'].value_counts()
			fluor5_tot = x[1]
		else:
			fluor5_tot = 0

		if any(rslt_df.randGFP5 == 2):
			x = rslt_df['randGFP5'].value_counts()
			nonfluor5_tot = x[2]
		else:
			nonfluor5_tot = 0

		# Creating list with variables we just calculated
		data = [[filename, steps, window_start, window_end, kernel_tot, fluor_tot, nonfluor_tot, fluor2_tot, nonfluor2_tot, fluor3_tot, nonfluor3_tot, fluor4_tot, nonfluor4_tot, fluor5_tot, nonfluor5_tot]]

		# Putting list into dataframe (which is just 1 row)
		data_df = pd.DataFrame(data=data,
							   columns='File Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor Fluor2 Nonfluor2 Fluor3 Nonfluor3 Fluor4 Nonfluor4 Fluor5 Nonfluor5'.split())

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
	temp_df = pd.DataFrame(columns='P-Value Comparison P2-Value Comparison2 P3-Value Comparison3 P4-Value Comparison4 P5-Value Comparison5'.split())

	# Beginning of loop to calculate chi squared for each window
	while index <= end_index:
		# Narrowing down df to single row with only rows for calculations
		single_row = kern_count_df.iloc[[index]]
		single_row = single_row.loc[:, 'Total_Kernels':'Nonfluor5']
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

		# Process repeated for each model
		fluor2 = single_row['Fluor2'].values[0]
		nonfluor2 = single_row['Nonfluor2'].values[0]

		chi2_stat = stats.chisquare([fluor2, nonfluor2], [expected, expected])
		pval2 = chi2_stat[1]

		if pval2 <= 0.05:
			p2_input = '< p = 0.05'
		else:
			p2_input = '> p = 0.05'

		fluor3 = single_row['Fluor3'].values[0]
		nonfluor3 = single_row['Nonfluor3'].values[0]

		chi3_stat = stats.chisquare([fluor3, nonfluor3], [expected, expected])
		pval3 = chi3_stat[1]

		if pval3 <= 0.05:
			p3_input = '< p = 0.05'
		else:
			p3_input = '> p = 0.05'

		fluor4 = single_row['Fluor4'].values[0]
		nonfluor4 = single_row['Nonfluor4'].values[0]

		chi4_stat = stats.chisquare([fluor4, nonfluor4], [expected, expected])
		pval4 = chi4_stat[1]

		if pval4 <= 0.05:
			p4_input = '< p = 0.05'
		else:
			p4_input = '> p = 0.05'

		fluor5 = single_row['Fluor5'].values[0]
		nonfluor5 = single_row['Nonfluor5'].values[0]

		chi5_stat = stats.chisquare([fluor5, nonfluor5], [expected, expected])
		pval5 = chi5_stat[1]

		if pval5 <= 0.05:
			p5_input = '< p = 0.05'
		else:
			p5_input = '> p = 0.05'

		# Putting variables into list
		data = [[pval, p_input, pval2, p2_input, pval3, p3_input, pval4, p4_input, pval5, p5_input]]
		# Putting list into dataframe (single row)
		data_df = pd.DataFrame(data=data, columns='P-Value Comparison P2-Value Comparison2 P3-Value Comparison3 P4-Value Comparison4 P5-Value Comparison5'.split())
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
# No normalization of x axis mean value but Normalized value added to df
def pval_plot(final_df, xml):
	# Hiding error message
	plt.rcParams.update({'figure.max_open_warning': 0})
	# Creating new column of 'window mean'
	col = final_df.loc[:, "Window_Start":"Window_End"]
	final_df['window_mean'] = col.mean(axis=1)
	# Normalizing window mean for x axis but not plotted..just put into df
	final_df['Normalized_Window_Mean'] = final_df.groupby('File')['window_mean'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

	# Calculating percent transmission for each model
	final_df['Percent_Transmission'] = final_df['Total_Fluor'] / final_df['Total_Kernels']
	final_df['Percent_Transmission2'] = final_df['Fluor2'] / final_df['Total_Kernels']
	final_df['Percent_Transmission3'] = final_df['Fluor3'] / final_df['Total_Kernels']
	final_df['Percent_Transmission4'] = final_df['Fluor4'] / final_df['Total_Kernels']
	final_df['Percent_Transmission5'] = final_df['Fluor5'] / final_df['Total_Kernels']

	# Counting number of rows in dataframe
	end_index = final_df.index[-1]

	# Calculating regression for each model
	reg_x = final_df['window_mean'].values
	reg_y = final_df['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	reg2_x = final_df['window_mean'].values
	reg2_y = final_df['Percent_Transmission2'].values
	slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(reg2_x, reg2_y)

	reg3_x = final_df['window_mean'].values
	reg3_y = final_df['Percent_Transmission3'].values
	slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(reg3_x, reg3_y)

	reg4_x = final_df['window_mean'].values
	reg4_y = final_df['Percent_Transmission4'].values
	slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(reg4_x, reg4_y)

	reg5_x = final_df['window_mean'].values
	reg5_y = final_df['Percent_Transmission5'].values
	slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(reg5_x, reg5_y)

	# Beginning of plotting... using line collections to plot segments that are
	# colored based on tuples (red or blue) for p values
	# One for each model
	segments = []
	colors = np.zeros(shape=(end_index * 5, 4))
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

	second_x = final_df['window_mean'].values
	second_y = final_df['Percent_Transmission2'].values
	second_z = final_df['P2-Value'].values

	for x3, x4, y3, y4, z3, z4 in zip(second_x, second_x[1:], second_y, second_y[1:], second_z, second_z[1:]):
		if z3 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z3 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments.append([(x3, y3), (x4, y4)])
		i += 1

	third_x = final_df['window_mean'].values
	third_y = final_df['Percent_Transmission3'].values
	third_z = final_df['P3-Value'].values

	for x5, x6, y5, y6, z5, z6 in zip(third_x, third_x[1:], third_y, third_y[1:], third_z, third_z[1:]):
		if z5 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z5 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments.append([(x5, y5), (x6, y6)])
		i += 1

	fourth_x = final_df['window_mean'].values
	fourth_y = final_df['Percent_Transmission4'].values
	fourth_z = final_df['P4-Value'].values

	for x7, x8, y7, y8, z7, z8 in zip(fourth_x, fourth_x[1:], fourth_y, fourth_y[1:], fourth_z, fourth_z[1:]):
		if z7 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z7 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments.append([(x7, y7), (x8, y8)])
		i += 1

	fifth_x = final_df['window_mean'].values
	fifth_y = final_df['Percent_Transmission5'].values
	fifth_z = final_df['P5-Value'].values

	for x9, x99, y9, y99, z9, z99 in zip(fifth_x, fifth_x[1:], fifth_y, fifth_y[1:], fifth_z, fifth_z[1:]):
		if z9 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z9 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments.append([(x9, y9), (x99, y99)])
		i += 1

	# Creating a line by putting segments together
	lc = mc.LineCollection(segments, colors=colors, linewidths=2)
	fig, ax = pl.subplots(figsize=(11.7, 8.27))
	# Adding line collection to axis
	ax.add_collection(lc)
	ax.autoscale()
	ax.margins(0.1)
	# Plotting regression line for each model
	plt.plot(reg_x, intercept + slope * reg_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])
	plt.plot(reg2_x, intercept2 + slope2 * reg2_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])
	plt.plot(reg3_x, intercept3 + slope3 * reg3_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])
	plt.plot(reg4_x, intercept4 + slope4 * reg4_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])
	plt.plot(reg5_x, intercept5 + slope5 * reg5_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])

	# Graph aesthetics, selecting size, adding titles, etc
	ax.set_xlim(np.min(second_x)-50, np.max(second_x)+50)
	ax.set_ylim(0, 1)
	plt.yticks(np.arange(0, 1, step=0.25))
	plt.figure(figsize=(11.7, 8.27))

	ax.set_title(xml[:-4]+' Model Plot', fontsize=30, fontweight='bold')
	ax.set_xlabel('Window Position (pixels)', fontsize=20, fontweight='bold')
	ax.set_ylabel('% GFP', fontsize=20, fontweight='bold')

	ax.set_facecolor('white')
	ax.yaxis.grid(color='grey')

	# Black boarder around graph
	ax.spines['bottom'].set_color('black')
	ax.spines['top'].set_color('black')
	ax.spines['right'].set_color('black')
	ax.spines['left'].set_color('black')

	# Creating legend for colored p value line
	red_patch = mpatches.Patch(color='red', label='> p = 0.05')
	blue_patch = mpatches.Patch(color='blue', label='< p = 0.05')
	plt.legend(handles=[red_patch, blue_patch], loc='center left', bbox_to_anchor=(1, 0.5))

	pv_plot = lc.get_figure()

	# Create directory to save plots
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Model_Transmission_plots/')
	# Sample_file_name
	sample_file_name = xml[:-4] + '_model.png'

	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)

	pv_plot.savefig(results_dir + sample_file_name, bbox_inches="tight")
	plt.close()

	# Creating temporary dataframe that will append to the meta df in main
	total_kernels = len(final_df.index)
	total_fluor = final_df['Total_Fluor'].sum()
	perc_trans = total_fluor/total_kernels

	all_data = [[xml[:-4], total_kernels, perc_trans, r_value ** 2, p_value, slope, slope2, intercept2, r_value2 ** 2, p_value2, slope3, intercept3, r_value3 ** 2, p_value3, slope4, intercept4, r_value4 ** 4, p_value4, slope5, intercept5, r_value5 ** 2, p_value5]]

	# Putting list into dataframe (which is just 1 row)
	data_df = pd.DataFrame(data=all_data, columns='File_Name Total_Kernels Percent_Transmission R-Squared P-Value Slope slope2 intercept2 r_value2 p_value2 slope3 intercept3 r_value3 p_value3 slope4 intercept4 r_value4 p_value4 slope5 intercept5 r_value5 p_value5'.split())

	return pv_plot, data_df

# Plots a line for percent transmission across the ear
# colored based on whether point is above or below p = 0.05
# Normalizes window mean x values
def pval_norm_plot(final_df, xml):
	# Hiding error message
	plt.rcParams.update({'figure.max_open_warning': 0})
	# Creating new column of 'window mean'
	col = final_df.loc[:, "Window_Start":"Window_End"]
	final_df['window_mean'] = col.mean(axis=1)
	# Normalizing window mean values for x axis
	final_df['Normalized_Window_Mean'] = final_df.groupby('File')['window_mean'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
	# Creating new column of percent transmission for each model
	final_df['Percent_Transmission'] = final_df['Total_Fluor'] / final_df['Total_Kernels']
	final_df['Percent_Transmission2'] = final_df['Fluor2'] / final_df['Total_Kernels']
	final_df['Percent_Transmission3'] = final_df['Fluor3'] / final_df['Total_Kernels']
	final_df['Percent_Transmission4'] = final_df['Fluor4'] / final_df['Total_Kernels']
	final_df['Percent_Transmission5'] = final_df['Fluor5'] / final_df['Total_Kernels']
	# Counting number of rows in dataframe
	end_index = final_df.index[-1]
	# Calculating regression for each model
	reg_x = final_df['Normalized_Window_Mean'].values
	reg_y = final_df['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	reg2_x = final_df['Normalized_Window_Mean'].values
	reg2_y = final_df['Percent_Transmission2'].values
	slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(reg2_x, reg2_y)

	reg3_x = final_df['Normalized_Window_Mean'].values
	reg3_y = final_df['Percent_Transmission3'].values
	slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(reg3_x, reg3_y)

	reg4_x = final_df['Normalized_Window_Mean'].values
	reg4_y = final_df['Percent_Transmission4'].values
	slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(reg4_x, reg4_y)

	reg5_x = final_df['Normalized_Window_Mean'].values
	reg5_y = final_df['Percent_Transmission5'].values
	slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(reg5_x, reg5_y)

	# Beginning of plotting... using line collections to plot segments that are
	# colored based on tuples (red or blue) for p values for each model
	segments = []
	colors = np.zeros(shape=(end_index * 5, 4))
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

	second_x = final_df['Normalized_Window_Mean'].values
	second_y = final_df['Percent_Transmission2'].values
	second_z = final_df['P2-Value'].values

	for x3, x4, y3, y4, z3, z4 in zip(second_x, second_x[1:], second_y, second_y[1:], second_z, second_z[1:]):
		if z3 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z3 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments.append([(x3, y3), (x4, y4)])
		i += 1

	third_x = final_df['Normalized_Window_Mean'].values
	third_y = final_df['Percent_Transmission3'].values
	third_z = final_df['P3-Value'].values

	for x5, x6, y5, y6, z5, z6 in zip(third_x, third_x[1:], third_y, third_y[1:], third_z, third_z[1:]):
		if z5 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z5 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments.append([(x5, y5), (x6, y6)])
		i += 1

	fourth_x = final_df['Normalized_Window_Mean'].values
	fourth_y = final_df['Percent_Transmission4'].values
	fourth_z = final_df['P4-Value'].values

	for x7, x8, y7, y8, z7, z8 in zip(fourth_x, fourth_x[1:], fourth_y, fourth_y[1:], fourth_z, fourth_z[1:]):
		if z7 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z7 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments.append([(x7, y7), (x8, y8)])
		i += 1

	fifth_x = final_df['Normalized_Window_Mean'].values
	fifth_y = final_df['Percent_Transmission5'].values
	fifth_z = final_df['P5-Value'].values

	for x9, x99, y9, y99, z9, z99 in zip(fifth_x, fifth_x[1:], fifth_y, fifth_y[1:], fifth_z, fifth_z[1:]):
		if z9 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z9 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments.append([(x9, y9), (x99, y99)])
		i += 1

	# Creating a line by putting segments together
	lc = mc.LineCollection(segments, colors=colors, linewidths=2)
	fig, ax = pl.subplots(figsize=(11.7, 8.27))
	ax.add_collection(lc)
	ax.autoscale()
	ax.margins(0.1)
	plt.plot(reg_x, intercept + slope * reg_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])
	plt.plot(reg2_x, intercept2 + slope2 * reg2_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])
	plt.plot(reg3_x, intercept3 + slope3 * reg3_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])
	plt.plot(reg4_x, intercept4 + slope4 * reg4_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])
	plt.plot(reg5_x, intercept5 + slope5 * reg5_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])

	# Settings for graph aesthetics, figure size, titles, etc
	ax.set_ylim(0, 1)
	plt.yticks(np.arange(0, 1, step=0.25))
	plt.figure(figsize=(11.7, 8.27))

	ax.set_title(xml[:-4]+' Model Plot', fontsize=30, fontweight='bold')
	ax.set_xlabel('Normalized Window Position (pixels)', fontsize=20, fontweight='bold')
	ax.set_ylabel('% GFP', fontsize=20, fontweight='bold')

	ax.set_facecolor('white')
	ax.yaxis.grid(color='grey')

	# Adding black box around graph
	ax.spines['bottom'].set_color('black')
	ax.spines['top'].set_color('black')
	ax.spines['right'].set_color('black')
	ax.spines['left'].set_color('black')

	# Creating legend for colored p value line
	red_patch = mpatches.Patch(color='red', label='> p = 0.05')
	blue_patch = mpatches.Patch(color='blue', label='< p = 0.05')
	plt.legend(handles=[red_patch, blue_patch], loc='center left', bbox_to_anchor=(1, 0.5))

	pv_plot = lc.get_figure()

	# Create directory to save plots
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Model_Norm_Transmission_plots/')
	# Sample_file_name
	sample_file_name = xml[:-4] + '_model.png'

	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)

	pv_plot.savefig(results_dir + sample_file_name, bbox_inches="tight")
	plt.close()

	# Creating temporary dataframe that will append to the meta df in main
	total_kernels = len(final_df.index)
	total_fluor = final_df['Total_Fluor'].sum()
	perc_trans = total_fluor/total_kernels

	all_data = [[xml[:-4], total_kernels, perc_trans, r_value ** 2, p_value, slope, slope2, intercept2, r_value2 ** 2, p_value2, slope3, intercept3, r_value3 ** 2, p_value3, slope4, intercept4, r_value4 ** 4, p_value4, slope5, intercept5, r_value5 ** 2, p_value5]]

	# Putting list into dataframe (which is just 1 row)
	data_df = pd.DataFrame(data=all_data, columns='File_Name Total_Kernels Percent_Transmission R-Squared P-Value Slope slope2 intercept2 r_value2 p_value2 slope3 intercept3 r_value3 p_value3 slope4 intercept4 r_value4 p_value4 slope5 intercept5 r_value5 p_value5'.split())

	return pv_plot, data_df

# Main function for running the whole script with argparse
# if - allows you to input xml file as first argument
# else - allows you to input directory of xml files as argument
def main():
	# Dataframe (saved to .txt file) for each file and their chisquare and regression stats
	meta_df = pd.DataFrame(columns='File_Name Total_Kernels Percent_Transmission R-Squared P-Value Slope slope2 intercept2 r_value2 p_value2 slope3 intercept3 r_value3 p_value3 slope4 intercept4 r_value4 p_value4 slope5 intercept5 r_value5 p_value5'.split())
	# Dataframe (saved to .txt file) for each file's name, window_mean, % transmission, pvalue, etc
	everything_df = pd.DataFrame(columns='File Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor Fluor2 Nonfluor2 Fluor3 Nonfluor3 Fluor4 Nonfluor4 Fluor5 Nonfluor5 P-Value Comparison P2-Value Comparison2 P3-Value Comparison3 P4-Value Comparison4 P5-Value Comparison5 window_mean Normalized_Window_Mean Percent_Transmission Percent_Transmission2 Percent_Transmission3 Percent_Transmission4 Percent_Transmission5'.split())

	if args.xml.endswith(".xml"):
		result, tree = check_xml_error(args.xml)
		if result == 'True':
			sys.exit('Program Exit')
		# Check xml error function
		print(f'Processing {args.xml}...')
		dataframe = parse_xml(args.xml, tree)
		dataframe2, ans1, ans2, ans3 = sliding_window(dataframe, args.width, args.step_size, args.xml)
		# Error checking for improper width or step size from sliding_window()
		if (ans1 == 'True') or (ans2 == 'True') or (ans3 == 'True'):
			sys.exit('Program Exit')
		chi_df = chisquare_test(dataframe2)
		# If -n present, normalize x coord on plots
		if args.n:
			trans_plot, end_df = pval_norm_plot(chi_df, args.xml)
		else:
			trans_plot, end_df = pval_plot(chi_df, args.xml)
		everything_df = everything_df.append(chi_df)
		everything_df = everything_df.reset_index(drop=True)
		meta_df = meta_df.append(end_df)
		meta_df = meta_df.reset_index(drop=True)
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
						dataframe = parse_xml(f, tree)
						dataframe2, ans1, ans2, ans3 = sliding_window(dataframe, args.width, args.step_size, filename)
						# Error checking for improper width or step size from sliding_window()
						if (ans1 == 'True') or (ans2 == 'True') or (ans3 == 'True'):
							continue
						chi_df = chisquare_test(dataframe2)
						# If -n present, normalize x coord on plots
						if args.n:
							trans_plot, end_df = pval_norm_plot(chi_df, filename)
						else:
							trans_plot, end_df = pval_plot(chi_df, filename)
						everything_df = everything_df.append(chi_df)
						everything_df = everything_df.reset_index(drop=True)
						meta_df = meta_df.append(end_df)
						meta_df = meta_df.reset_index(drop=True)
	# Saving both data frames to .txt file
	everything_df.to_csv('everything_model_df.txt', sep='\t')
	meta_df.to_csv('meta_model_df.txt', sep='\t')
	print('Process Complete!')

if __name__ == '__main__':
	main()