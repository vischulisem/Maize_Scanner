#!/usr/local/bin/python3

#This script contains functions to extract XML kernel coordinates and put them into a dataframe
#Next coordinates are plotted on a scatter plot to show the fluorescent and nonfluorescent kernel locations
#Finally, Sliding_Window function generates a new dataframe based on desired width and steps of window
# # and generates a plot comparing perecent transmission across different windows on the ear

# # # Input arguments are an XML file, width, and steps
# # # # Output is a scatter plot saved to a folder containing 'XML_filename'.png

import sys
import os
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from pylab import rcParams
from scipy import stats
import pylab as pl
from matplotlib import collections  as mc


#setting up argparse arguments
parser = argparse.ArgumentParser(description='Given XML file, width, and steps, returns scatterplot')
parser.add_argument('xml', metavar='', help='Input XML filename.', type=str)
parser.add_argument('width', metavar='', help='Width in pixels for the length of the window.', type=int)
parser.add_argument('step_size', metavar='', help='Steps in pixels for window movement.', type=int)
args = parser.parse_args()


def check_xml_error( input_xml ):
	# Make element tree for object
	tree = ET.parse(input_xml)

	# Getting the root of the tree
	root = tree.getroot()

	# Pulling out the name of the image
	image_name_string = (root[0][0].text)

	# Assigning types other than fluorescent and nonfluor in order to
	# # exit program if list is present
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


	# #checking if anything exists in other types
	if (count_4 > 1) or (count_5 > 1) or (count_6> 1) or (count_7 > 1) or (count_8 > 1):
		print(f'ERROR: {image_name_string} skipped...contains unknown type.')
		result = 'True'
	else:
		result = 'False'

	return result, tree



def parse_xml(input_xml, tree):

	# Make element tree for object
	#tree = ET.parse(input_xml)

	# Getting the root of the tree
	root = tree.getroot()

	# Pulling out the name of the image
	image_name_string = (root[0][0].text)

	# Pulling out the fluorescent and non-fluorescent children
	fluorescent = root[1][1]
	nonfluorescent = root[1][2]

	# Setting up some empty lists to move the coordinates from the xml into
	fluor_x = []
	fluor_y = []
	nonfluor_x = []
	nonfluor_y = []

	# # Getting the coordinates of the fluorescent kernels
	for child in fluorescent:
		if child.tag == 'Marker':
			fluor_x.append(child.find('MarkerX').text)
			fluor_y.append(child.find('MarkerY').text)

	# # Getting the coordinates of the non-fluorescent kernels
	for child in nonfluorescent:
		if child.tag == 'Marker':
			nonfluor_x.append(child.find('MarkerX').text)
			nonfluor_y.append(child.find('MarkerY').text)

	# # Creating the repeating 'type' column values
	fluor_type = 'Fluorescent'
	nonfluor_type = 'Non-Fluorescent'

	# Putting together the results for output in [file name, type, x coord, y coord] format
	fluor_coord = np.column_stack(([image_name_string] * len(fluor_x), [fluor_type] * len(fluor_x), fluor_x, fluor_y))
	nonfluor_coord = np.column_stack(
		([image_name_string] * len(nonfluor_x), [nonfluor_type] * len(nonfluor_x), nonfluor_x, nonfluor_y))

	# # Stacking the fluor and nonfluor arrays on top of eachother
	combined_array = np.vstack((fluor_coord, nonfluor_coord))

	# #Importing np.array into pandas dataframe and converting coordinate values from objects to integers
	df = pd.DataFrame(data=combined_array, columns='File Type X-Coordinate Y-Coordinate'.split())

	df['randGFP1'] = np.random.randint(1, 3, df.shape[0])
	df['randGFP2'] = np.random.randint(1, 3, df.shape[0])
	df['randGFP3'] = np.random.randint(1, 3, df.shape[0])
	df['randGFP4'] = np.random.randint(1, 3, df.shape[0])
	df['randGFP5'] = np.random.randint(1, 3, df.shape[0])

	df['X-Coordinate'] = df['X-Coordinate'].astype(np.int64)
	df['Y-Coordinate'] = df['Y-Coordinate'].astype(np.int64)

	# #overall ear stats1
	# overall_1kernel_total = int(len(df.index))
	# overall_1expected = overall_1kernel_total * 0.5
	#
	# if any(df.randGFP1 == 1):
	# 	x = df['randGFP1'].value_counts()
	# 	overall_1fluor_tot = x[1]
	# else:
	# 	overall_1fluor_tot = 0
	#
	# overall_1perc_trans = overall_1fluor_tot/overall_1kernel_total
	#
	# if any(df.randGFP1 == 2):
	# 	x = df['randGFP1'].value_counts()
	# 	overall_1nonfluor_tot = x[2]
	# else:
	# 	overall_1nonfluor_tot = 0
	#
	# chi_1stat = stats.chisquare([overall_1fluor_tot, overall_1nonfluor_tot], [overall_1expected, overall_1expected])
	# overall_1pval = chi_1stat[1]
	#
	#
	# # overall ear stats2
	# overall_2kernel_total = int(len(df.index))
	# overall_2expected = overall_2kernel_total * 0.5
	#
	# if any(df.randGFP2 == 1):
	# 	x = df['randGFP2'].value_counts()
	# 	overall_2fluor_tot = x[1]
	# else:
	# 	overall_2fluor_tot = 0
	#
	# overall_2perc_trans = overall_2fluor_tot / overall_2kernel_total
	#
	# if any(df.randGFP2 == 2):
	# 	x = df['randGFP2'].value_counts()
	# 	overall_2nonfluor_tot = x[2]
	# else:
	# 	overall_2nonfluor_tot = 0
	#
	# chi_2stat = stats.chisquare([overall_2fluor_tot, overall_2nonfluor_tot], [overall_2expected, overall_2expected])
	# overall_2pval = chi_2stat[1]
	#
	#
	# # overall ear stats3
	# overall_3kernel_total = int(len(df.index))
	# overall_3expected = overall_3kernel_total * 0.5
	#
	# if any(df.randGFP3 == 1):
	# 	x = df['randGFP3'].value_counts()
	# 	overall_3fluor_tot = x[1]
	# else:
	# 	overall_3fluor_tot = 0
	#
	# overall_3perc_trans = overall_3fluor_tot / overall_3kernel_total
	#
	# if any(df.randGFP3 == 2):
	# 	x = df['randGFP3'].value_counts()
	# 	overall_3nonfluor_tot = x[2]
	# else:
	# 	overall_3nonfluor_tot = 0
	#
	# chi_3stat = stats.chisquare([overall_3fluor_tot, overall_3nonfluor_tot], [overall_3expected, overall_3expected])
	# overall_3pval = chi_3stat[1]
	#
	# # overall ear stats4
	# overall_4kernel_total = int(len(df.index))
	# overall_4expected = overall_4kernel_total * 0.5
	#
	# if any(df.randGFP4 == 1):
	# 	x = df['randGFP4'].value_counts()
	# 	overall_4fluor_tot = x[1]
	# else:
	# 	overall_4fluor_tot = 0
	#
	# overall_4perc_trans = overall_4fluor_tot / overall_4kernel_total
	#
	# if any(df.randGFP4 == 2):
	# 	x = df['randGFP4'].value_counts()
	# 	overall_4nonfluor_tot = x[2]
	# else:
	# 	overall_4nonfluor_tot = 0
	#
	# chi_4stat = stats.chisquare([overall_4fluor_tot, overall_4nonfluor_tot], [overall_4expected, overall_4expected])
	# overall_4pval = chi_4stat[1]
	#
	# # overall ear stats5
	# overall_5kernel_total = int(len(df.index))
	# overall_5expected = overall_5kernel_total * 0.5
	#
	# if any(df.randGFP5 == 1):
	# 	x = df['randGFP5'].value_counts()
	# 	overall_5fluor_tot = x[1]
	# else:
	# 	overall_5fluor_tot = 0
	#
	# overall_5perc_trans = overall_5fluor_tot / overall_5kernel_total
	#
	# if any(df.randGFP5 == 2):
	# 	x = df['randGFP5'].value_counts()
	# 	overall_5nonfluor_tot = x[2]
	# else:
	# 	overall_5nonfluor_tot = 0
	#
	# chi_5stat = stats.chisquare([overall_5fluor_tot, overall_5nonfluor_tot], [overall_5expected, overall_5expected])
	# overall_5pval = chi_5stat[1]

	return df


# # End of function

# Generating plot of coordinate values on ear of fluor and nonfluor
#def make_scatter(df):
	#sns.set(rc={'figure.figsize': (9, 2.5)})
	#ax = sns.scatterplot("X-Coordinate", "Y-Coordinate", hue="Type", data=df, palette='Set1')
	#handles, labels = ax.get_legend_handles_labels()
	#l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	#ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
	#plt.axis('equal')
	#figure = ax.get_figure()
	# figure.savefig("coord_plot.png")
	# my_plot = plt.show()

	#return figure


def sliding_window(df, w, s, filename):
	# # sort x values from small to big
	df.sort_values(by=['X-Coordinate'], inplace=True)
	df = df.reset_index(drop=True)

	# #Choosing starting point for window with value for x
	start_x = df["X-Coordinate"].head(1)
	int_start_x = int(start_x)

	# # setting up w and s as integers
	int_w = int(w)
	int_s = int(s)

	# #Defining end of window
	end_x = int_start_x + int_w

	# #Defining steps for window
	steps = int_s

	# # creating empty dataframe
	kern_count_df = pd.DataFrame(
		columns='File Step_Size Window_Start Window_End Total_Kernels Fluor1 Fluor2 Fluor3 Fluor4 Fluor5 NonFluor1 NonFluor2 NonFluor3 NonFluor4 NonFluor5'.split())

	#Assigning variable to final x coordinate in dataframe
	final_x_coord = df["X-Coordinate"].tail(1)
	int_fxc = int(final_x_coord)

	# # Creating error messages for too small or too big input width or steps
	adj_step_fxc = int_fxc * 0.25

	if (end_x >= int_fxc):
		print(f'For {filename} - Width of window is too large, enter smaller width value.')
		ans1 = 'True'
	else:
		ans1 = 'False'

	if (int_s >= adj_step_fxc):
		print(f'For {filename} - Step size is too large, enter smaller value.')
		ans2 = 'True'
	else:
		ans2 = 'False'

	# # Beginning of sliding window to scan ear and output new dataframe called kern_count_df
	while end_x <= int_fxc:

		#Creating smaller window df based on original df
		rslt_df = df[(df['X-Coordinate'] >= int_start_x) & (df['X-Coordinate'] <= end_x)]

		#Total kernels in window
		kernel_tot = len(rslt_df.index)


		#Error message if there are no kernels in window
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
			x = df['randGFP1'].value_counts()
			o_1fluor_tot = x[1]
		else:
			o_1fluor_tot = 0

		if any(rslt_df.randGFP2 == 1):
			x = df['randGFP2'].value_counts()
			o_2fluor_tot = x[1]
		else:
			o_2fluor_tot = 0

		if any(rslt_df.randGFP3 == 1):
			x = df['randGFP3'].value_counts()
			o_3fluor_tot = x[1]
		else:
			o_3fluor_tot = 0

		if any(rslt_df.randGFP4 == 1):
			x = df['randGFP4'].value_counts()
			o_4fluor_tot = x[1]
		else:
			o_4fluor_tot = 0

		if any(rslt_df.randGFP5 == 1):
			x = df['randGFP5'].value_counts()
			o_5fluor_tot = x[1]
		else:
			o_5fluor_tot = 0

		if any(rslt_df.randGFP1 == 2):
			x = df['randGFP1'].value_counts()
			o_1nonfluor_tot = x[2]
		else:
			o_1nonfluor_tot = 0

		if any(rslt_df.randGFP2 == 2):
			x = df['randGFP2'].value_counts()
			o_2nonfluor_tot = x[2]
		else:
			o_2nonfluor_tot = 0

		if any(rslt_df.randGFP3 == 2):
			x = df['randGFP3'].value_counts()
			o_3nonfluor_tot = x[2]
		else:
			o_3nonfluor_tot = 0

		if any(rslt_df.randGFP4 == 2):
			x = df['randGFP4'].value_counts()
			o_4nonfluor_tot = x[2]
		else:
			o_4nonfluor_tot = 0

		if any(rslt_df.randGFP5 == 2):
			x = df['randGFP5'].value_counts()
			o_5nonfluor_tot = x[2]
		else:
			o_5nonfluor_tot = 0

		#creating list with variables we just calculated
		data = [[filename, steps, window_start, window_end, kernel_tot, o_1fluor_tot, o_2fluor_tot, o_3fluor_tot, o_4fluor_tot, o_5fluor_tot, o_1nonfluor_tot, o_2nonfluor_tot, o_3nonfluor_tot, o_4nonfluor_tot, o_5nonfluor_tot]]

		#putting list into dataframe (which is just 1 row)
		data_df = pd.DataFrame(data=data,
							   columns='File Step_Size Window_Start Window_End Total_Kernels Fluor1 Fluor2 Fluor3 Fluor4 Fluor5 NonFluor1 NonFluor2 NonFluor3 NonFluor4 NonFluor5'.split())

		#appending data_df to kern_count_df (1 row added each time)
		kern_count_df = kern_count_df.append(data_df)

		#shifting window based on stepsize
		int_start_x = int_start_x + steps
		end_x = end_x + steps

	#resetting index
	kern_count_df = kern_count_df.reset_index(drop=True)
	cols = kern_count_df.columns.drop(['File'])
	kern_count_df[cols] = kern_count_df[cols].apply(pd.to_numeric)

	ans3 = 'Neither'

	return kern_count_df, ans1, ans2, ans3

#function for plotting total kernels vs average window position
def tot_kern_scatter ( kern_count_df ):
	col = kern_count_df.loc[: , "Window_Start":"Window_End"]
	kern_count_df['window_mean'] = col.mean(axis=1)

	kern_tot_scatter = sns.scatterplot("window_mean", "Total_Kernels", data=kern_count_df, palette='Set1')
	tot_kern_figure = kern_tot_scatter.get_figure()
	tot_kern_figure.savefig("tot_kern_figure.png")
	my_plot = plt.show()

	return tot_kern_figure

#function for plotting percent transmission to average window position
# # plots are saved by file name.png and put into new directory called
# # # transmission_plots
def transmission_scatter ( kern_count_df, xml ):
	# sets parameters for seaborn plots
	sns.set_style("white")

	#calculating average window position
	col = kern_count_df.loc[:, "Window_Start":"Window_End"]
	kern_count_df['window_mean'] = col.mean(axis=1)

	#calculating percent transmission
	kern_count_df['Percent_Transmission'] = kern_count_df['Total_Fluor']/kern_count_df['Total_Kernels']

	#creating plot
	transmission_plot = sns.lineplot(x="window_mean", y="Percent_Transmission", data=kern_count_df, linewidth=5)
	sns.set(rc={'figure.figsize':(11.7,8.27)})
	# plt.gcf().subplots_adjust(bottom=0.3)
	plt.ylim(0, 1)
	transmission_plot.yaxis.grid(True)


	plt.title(xml[:-4]+' Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"

	transmission_figure = transmission_plot.get_figure()

	#create directory to save plots
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Model_Transmission_plots/')
	#sample_file_name
	sample_file_name = xml[:-4]+'_model.png'

	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)

	transmission_figure.savefig(results_dir + sample_file_name, bbox_inches="tight")
	plt.close()

	return transmission_figure



def chisquare_test ( kern_count_df ):

	index = 0
	end_index = kern_count_df.index[-1]
	temp_df = pd.DataFrame(columns='P-Value1 Comparison1 P-Value2 Comparison2 P-Value3 Comparison3 P-Value4 Comparison4 P-Value5 Comparison5'.split())

	while index <= end_index:

		single_row = kern_count_df.iloc[[index]]
		single_row = single_row.loc[:, 'Total_Kernels':'NonFluor5']

		expected = single_row['Total_Kernels'].values[0] * 0.5
		fluor1 = single_row['Fluor1'].values[0]
		nonfluor1 = single_row['NonFluor1'].values[0]

		chi_1stat = stats.chisquare([fluor1, nonfluor1], [expected, expected])
		pval1 = chi_1stat[1]

		if pval1 <= 0.05:
			p_1input = '< p = 0.05'
		else:
			p_1input = '> p = 0.05'

		fluor2 = single_row['Fluor2'].values[0]
		nonfluor2 = single_row['NonFluor2'].values[0]

		chi_2stat = stats.chisquare([fluor2, nonfluor2], [expected, expected])
		pval2 = chi_2stat[1]

		if pval2 <= 0.05:
			p_2input = '< p = 0.05'
		else:
			p_2input = '> p = 0.05'

		fluor3 = single_row['Fluor3'].values[0]
		nonfluor3 = single_row['NonFluor3'].values[0]

		chi_3stat = stats.chisquare([fluor3, nonfluor3], [expected, expected])
		pval3 = chi_3stat[1]

		if pval3 <= 0.05:
			p_3input = '< p = 0.05'
		else:
			p_3input = '> p = 0.05'

		fluor4 = single_row['Fluor4'].values[0]
		nonfluor4 = single_row['NonFluor4'].values[0]

		chi_4stat = stats.chisquare([fluor4, nonfluor4], [expected, expected])
		pval4 = chi_4stat[1]

		if pval4 <= 0.05:
			p_4input = '< p = 0.05'
		else:
			p_4input = '> p = 0.05'

		fluor5 = single_row['Fluor5'].values[0]
		nonfluor5 = single_row['NonFluor5'].values[0]

		chi_5stat = stats.chisquare([fluor5, nonfluor5], [expected, expected])
		pval5 = chi_5stat[1]

		if pval5 <= 0.05:
			p_5input = '< p = 0.05'
		else:
			p_5input = '> p = 0.05'

		data = [[pval1, p_1input, pval2, p_2input, pval3, p_3input, pval4, p_4input, pval5, p_5input]]
		data_df = pd.DataFrame(data=data, columns='P-Value1 Comparison1 P-Value2 Comparison2 P-Value3 Comparison3 P-Value4 Comparison4 P-Value5 Comparison5'.split())
		temp_df = temp_df.append(data_df)

		index = index + 1
	temp_df = temp_df.reset_index(drop=True)
	final_df = pd.concat([kern_count_df, temp_df], axis=1, sort=False)
	final_df = final_df.reset_index(drop=True)

	return final_df

def pval_plot( final_df, xml):
	plt.rcParams.update({'figure.max_open_warning': 0})

	col = final_df.loc[:, "Window_Start":"Window_End"]
	final_df['window_mean'] = col.mean(axis=1)

	final_df['Percent_1Transmission'] = final_df['Fluor1'] / final_df['Total_Kernels']
	final_df['Percent_2Transmission'] = final_df['Fluor2'] / final_df['Total_Kernels']
	final_df['Percent_3Transmission'] = final_df['Fluor3'] / final_df['Total_Kernels']
	final_df['Percent_4Transmission'] = final_df['Fluor4'] / final_df['Total_Kernels']
	final_df['Percent_5Transmission'] = final_df['Fluor5'] / final_df['Total_Kernels']

	end_index = final_df.index[-1]

	reg_x = final_df['window_mean'].values
	reg_1y = final_df['Percent_1Transmission'].values
	reg_2y = final_df['Percent_2Transmission'].values
	reg_3y = final_df['Percent_3Transmission'].values
	reg_4y = final_df['Percent_4Transmission'].values
	reg_5y = final_df['Percent_5Transmission'].values

	slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(reg_x, reg_1y)
	slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(reg_x, reg_2y)
	slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(reg_x, reg_3y)
	slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(reg_x, reg_4y)
	slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(reg_x, reg_5y)

	rsq1 = r_value1 ** 2
	rsq2 = r_value2 ** 2
	rsq3 = r_value3 ** 2
	rsq4 = r_value4 ** 2
	rsq5 = r_value5 ** 2

	segments1 = []
	colors = np.zeros(shape=(end_index, 4))
	x = final_df['window_mean'].values
	y = final_df['Percent_1Transmission'].values
	z = final_df['P-Value1'].values
	i = 0

	for x1, x2, y1, y2, z1, z2 in zip(x, x[1:], y, y[1:], z, z[1:]):
		if z1 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z1 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments1.append([(x1, y1), (x2, y2)])
		i += 1

	lc1 = mc.LineCollection(segments1, colors=colors, linewidths=2)

	segments2 = []
	colors = np.zeros(shape=(end_index, 4))
	x = final_df['window_mean'].values
	yy = final_df['Percent_2Transmission'].values
	zz = final_df['P-Value2'].values
	i = 0

	for x1, x2, y1, y2, z1, z2 in zip(x, x[1:], yy, yy[1:], zz, zz[1:]):
		if z1 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z1 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments2.append([(x1, y1), (x2, y2)])
		i += 1

	lc2 = mc.LineCollection(segments2, colors=colors, linewidths=2)

	segments3 = []
	colors = np.zeros(shape=(end_index, 4))
	x = final_df['window_mean'].values
	yyy = final_df['Percent_3Transmission'].values
	zzz = final_df['P-Value3'].values
	i = 0

	for x1, x2, y1, y2, z1, z2 in zip(x, x[1:], yyy, yyy[1:], zzz, zzz[1:]):
		if z1 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z1 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments3.append([(x1, y1), (x2, y2)])
		i += 1

	lc3 = mc.LineCollection(segments3, colors=colors, linewidths=2)

	segments4 = []
	colors = np.zeros(shape=(end_index, 4))
	x = final_df['window_mean'].values
	yyyy = final_df['Percent_4Transmission'].values
	zzzz = final_df['P-Value4'].values
	i = 0

	for x1, x2, y1, y2, z1, z2 in zip(x, x[1:], yyyy, yyyy[1:], zzzz, zzzz[1:]):
		if z1 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z1 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments4.append([(x1, y1), (x2, y2)])
		i += 1

	lc4 = mc.LineCollection(segments4, colors=colors, linewidths=2)

	segments5 = []
	colors = np.zeros(shape=(end_index, 4))
	x = final_df['window_mean'].values
	yyyyy = final_df['Percent_5Transmission'].values
	zzzzz = final_df['P-Value5'].values
	i = 0

	for x1, x2, y1, y2, z1, z2 in zip(x, x[1:], yyyyy, yyyyy[1:], zzzzz, zzzzz[1:]):
		if z1 > 0.05:
			colors[i] = tuple([1, 0, 0, 1])
		elif z1 <= 0.05:
			colors[i] = tuple([0, 0, 1, 1])
		else:
			colors[i] = tuple([0, 1, 0, 1])
		segments5.append([(x1, y1), (x2, y2)])
		i += 1

	lc5 = mc.LineCollection(segments5, colors=colors, linewidths=2)

	fig, ax = pl.subplots(figsize=(11.7,8.27))
	ax.add_collection(lc1)
	ax.add_collection(lc2)
	ax.add_collection(lc3)
	ax.add_collection(lc4)
	ax.add_collection(lc5)
	ax.autoscale()
	ax.margins(0.1)
	plt.plot(reg_x, intercept1 + slope1 * reg_x, 'r', color='black', linewidth=3, dashes=[5, 3])
	plt.plot(reg_x, intercept2 + slope2 * reg_x, 'r', color='black', linewidth=3, dashes=[5, 3])
	plt.plot(reg_x, intercept3 + slope3 * reg_x, 'r', color='black', linewidth=3, dashes=[5, 3])
	plt.plot(reg_x, intercept4 + slope4 * reg_x, 'r', color='black', linewidth=3, dashes=[5, 3])
	plt.plot(reg_x, intercept5 + slope5 * reg_x, 'r', color='black', linewidth=3, dashes=[5, 3])

	ax.set_xlim(np.min(x)-50, np.max(x)+50)
	ax.set_ylim(0, 1)
	plt.yticks(np.arange(0, 1, step=0.25))
	plt.figure(figsize=(11.7, 8.27))


	ax.set_title(xml[:-4]+' Model Plot', fontsize=30, fontweight='bold')
	ax.set_xlabel('Window Position (pixels)', fontsize=20, fontweight='bold')
	ax.set_ylabel('% GFP', fontsize=20, fontweight='bold')

	ax.set_facecolor('white')
	ax.yaxis.grid(color='grey')

	ax.spines['bottom'].set_color('black')
	ax.spines['top'].set_color('black')
	ax.spines['right'].set_color('black')
	ax.spines['left'].set_color('black')

	red_patch = mpatches.Patch(color='red', label='> p = 0.05')
	blue_patch = mpatches.Patch(color='blue', label='< p = 0.05')
	ax.legend(handles=[red_patch, blue_patch], loc='center left', bbox_to_anchor=(1, 0.5))
	#
	# num_weird_trans = len(final_df[final_df['Comparison'] == '< p = 0.05'])
	# num_tkern = int(len(final_df))
	#
	# window_stat = num_weird_trans / num_tkern
	# window_stat = round(window_stat, 3)
	#
	# overall_kernel_total = round(overall_kernel_total, 3)
	# overall_perc_trans = round(overall_perc_trans, 3)
	# overall_pval = '{:0.3e}'.format(overall_pval)
	# slope = round(slope, 5)
	# intercept = round(intercept, 3)
	# rsq = round(rsq, 3)
	# p_value = '{:0.3e}'.format(p_value)
	#
	#
	#
	# textstr = '\n'.join((f'Overall Total Kernels = {overall_kernel_total}',
	# 					 f'Overall Percent Transmission = {overall_perc_trans}',
	# 					 f'Overall ChiSquared P-Value = {overall_pval}',
	# 					 f'% Windows not 0.5 Transmission = {window_stat}',
	# 					 f'Regression Slope = {slope}',
	# 					 f'Regression Intercept = {intercept}',
	# 					 f'Regression R-squared = {rsq}',
	# 					 f'Regression P-Value = {p_value}'))
	#
	# ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, fontweight='bold',
	# 		verticalalignment='top', bbox={'facecolor':'white', 'alpha':1, 'pad':10, 'edgecolor':'black'})

	pv_plot = lc5.get_figure()

	# create directory to save plots
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Model_Transmission_plots/')
	# sample_file_name
	sample_file_name = xml[:-4] + '_model.png'

	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)

	pv_plot.savefig(results_dir + sample_file_name, bbox_inches="tight")
	plt.close()


# Creating temporary dataframe that will append to the meta df in main

	total_kernels = len(final_df.index)
	total_1fluor = final_df['Fluor1'].sum()
	total_2fluor = final_df['Fluor2'].sum()
	total_3fluor = final_df['Fluor3'].sum()
	total_4fluor = final_df['Fluor4'].sum()
	total_5fluor = final_df['Fluor5'].sum()

	perc_1trans = total_1fluor/total_kernels
	perc_2trans = total_2fluor / total_kernels
	perc_3trans = total_3fluor / total_kernels
	perc_4trans = total_4fluor / total_kernels
	perc_5trans = total_5fluor / total_kernels

	all_data = [[xml[:-4], total_kernels, perc_1trans, rsq1, p_value1, slope1, perc_2trans, rsq2, p_value2, slope2, perc_3trans, rsq3, p_value3, slope3, perc_4trans, rsq4, p_value4, slope4, perc_5trans, rsq5, p_value5, slope5]]

	# putting list into dataframe (which is just 1 row)
	data_df = pd.DataFrame(data=all_data, columns='File_Name Total_Kernels Percent_Transmission1 R-Squared1 P-Value1 Slope1 Percent_Transmission2 R-Squared2 P-Value2 Slope2 Percent_Transmission3 R-Squared3 P-Value3 Slope3 Percent_Transmission4 R-Squared4 P-Value4 Slope4 Percent_Transmission5 R-Squared5 P-Value5 Slope5'.split())

	return pv_plot, data_df





#Main function for running the whole script with argparse
# # if - allows you to input xml file as first argument
# # # else - allows you to input directory of xml files as argument
def main():
	meta_df = pd.DataFrame(columns='File_Name Total_Kernels Percent_Transmission1 R-Squared1 P-Value1 Slope1 Percent_Transmission2 R-Squared2 P-Value2 Slope2 Percent_Transmission3 R-Squared3 P-Value3 Slope3 Percent_Transmission4 R-Squared4 P-Value4 Slope4 Percent_Transmission5 R-Squared5 P-Value5 Slope5'.split())
	everything_df = pd.DataFrame(columns='File Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor P-Value Comparison window_mean Percent_Transmission'.split())

	if args.xml.endswith(".xml"):
		result, tree = check_xml_error(args.xml)
		if result == 'True':
			sys.exit('Program Exit')
		# check xml error fun
		print(f'Processing {args.xml}...')
		dataframe, overall_1kernel_total, overall_1perc_trans, overall_1pval, overall_2kernel_total, overall_2perc_trans, overall_2pval, overall_3kernel_total, overall_3perc_trans, overall_3pval, overall_4kernel_total, overall_4perc_trans, overall_4pval, overall_5kernel_total, overall_5perc_trans, overall_5pval = parse_xml(args.xml, tree)
		dataframe2, ans1, ans2, ans3 = sliding_window(dataframe, args.width, args.step_size, args.xml)
		if (ans1 == 'True') or (ans2 == 'True') or (ans3 == 'True'):
			sys.exit('Program Exit')
		chi_df = chisquare_test(dataframe2)
		trans_plot, end_df = pval_plot(chi_df, args.xml, overall_1kernel_total, overall_1perc_trans, overall_1pval, overall_2kernel_total, overall_2perc_trans, overall_2pval, overall_3kernel_total, overall_3perc_trans, overall_3pval, overall_4kernel_total, overall_4perc_trans, overall_4pval, overall_5kernel_total, overall_5perc_trans, overall_5pval)
		meta_df = meta_df.append(end_df)
		meta_df = meta_df.reset_index(drop=True)
		# meta_df.to_csv('meta_df.txt', sep='\t')

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
						if (ans1 == 'True') or (ans2 == 'True') or (ans3 == 'True'):
							continue
						chi_df = chisquare_test(dataframe2)
						trans_plot, end_df = pval_plot(chi_df, filename)

						everything_df = everything_df.append(chi_df)
						everything_df = everything_df.reset_index(drop=True)
						meta_df = meta_df.append(end_df)
						meta_df = meta_df.reset_index(drop=True)

	everything_df.to_csv('model_everything_df.txt', sep='\t')
	meta_df.to_csv('model_meta_df.txt', sep='\t')
	print('Process Complete!')



if __name__ == '__main__':
	main()

