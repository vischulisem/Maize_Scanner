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
	df['X-Coordinate'] = df['X-Coordinate'].astype(np.int64)
	df['Y-Coordinate'] = df['Y-Coordinate'].astype(np.int64)


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
		columns='File Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor'.split())

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

		#creating list with variables we just calculated
		data = [[filename, steps, window_start, window_end, kernel_tot, fluor_tot, nonfluor_tot]]

		#putting list into dataframe (which is just 1 row)
		data_df = pd.DataFrame(data=data,
							   columns='File Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor'.split())

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
	results_dir = os.path.join(script_dir, 'Transmission_plots/')
	#sample_file_name
	sample_file_name = xml[:-4]+'.png'

	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)

	transmission_figure.savefig(results_dir + sample_file_name, bbox_inches="tight")
	plt.close()

	return transmission_figure



def chisquare_test ( kern_count_df ):

	index = 0
	end_index = kern_count_df.index[-1]
	temp_df = pd.DataFrame(columns='P-Value Comparison'.split())

	while index <= end_index:

		single_row = kern_count_df.iloc[[index]]
		single_row = single_row.loc[:, 'Total_Kernels':'Total_NonFluor']

		expected = single_row['Total_Kernels'].values[0] * 0.5
		fluor = single_row['Total_Fluor'].values[0]
		nonfluor = single_row['Total_NonFluor'].values[0]

		chi_stat = stats.chisquare([fluor, nonfluor], [expected, expected])
		pval = chi_stat[1]

		if pval <= 0.05:
			p_input = '< p = 0.05'
		else:
			p_input = '> p = 0.05'

		data = [[pval, p_input]]
		data_df = pd.DataFrame(data=data, columns='P-Value Comparison'.split())
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

	final_df['Percent_Transmission'] = final_df['Total_Fluor'] / final_df['Total_Kernels']
	end_index = final_df.index[-1]

	reg_x = final_df['window_mean'].values
	reg_y = final_df['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

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

	lc = mc.LineCollection(segments, colors=colors, linewidths=2)
	fig, ax = pl.subplots(figsize=(11.7,8.27))
	ax.add_collection(lc)
	ax.autoscale()
	ax.margins(0.1)
	plt.plot(reg_x, intercept + slope * reg_x, 'r', label='fitted line', color='black', linewidth=3, dashes=[5, 3])

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

	red_patch = mpatches.Patch(color='red', label='> p = 0.05')
	blue_patch = mpatches.Patch(color='blue', label='< p = 0.05')
	ax.legend(handles=[red_patch, blue_patch], loc='center left', bbox_to_anchor=(1, 0.5))

	pv_plot = lc.get_figure()

	# create directory to save plots
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Transmission_plots/')
	# sample_file_name
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

	# putting list into dataframe (which is just 1 row)
	data_df = pd.DataFrame(data=all_data, columns='File_Name Total_Kernels Percent_Transmission R-Squared P-Value Slope'.split())

	return pv_plot, data_df





#Main function for running the whole script with argparse
# # if - allows you to input xml file as first argument
# # # else - allows you to input directory of xml files as argument
def main():
	meta_df = pd.DataFrame(columns='File_Name Total_Kernels Percent_Transmission R-Squared P-Value Slope'.split())
	everything_df = pd.DataFrame(columns='File Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor P-Value Comparison window_mean Percent_Transmission'.split())

	if args.xml.endswith(".xml"):
		result, tree = check_xml_error(args.xml)
		if result == 'True':
			sys.exit('Program Exit')
		# check xml error fun
		print(f'Processing {args.xml}...')
		dataframe = parse_xml(args.xml, tree)
		dataframe2, ans1, ans2, ans3 = sliding_window(dataframe, args.width, args.step_size, args.xml)
		if (ans1 == 'True') or (ans2 == 'True') or (ans3 == 'True'):
			sys.exit('Program Exit')
		chi_df = chisquare_test(dataframe2)
		trans_plot, end_df = pval_plot(chi_df, args.xml)
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

	everything_df.to_csv('everything_df.txt', sep='\t')
	meta_df.to_csv('meta_df.txt', sep='\t')
	print('Process Complete!')



if __name__ == '__main__':
	main()

