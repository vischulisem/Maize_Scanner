#!/usr/local/bin/python3

#This script contains functions to extract XML kernel coordinates and put them into a dataframe
#Next coordinates are plotted on a scatter plot to show the fluorescent and nonfluorescent kernel locations
#Finally, Sliding_Window function generates a new dataframe based on desired width and steps of window
# # and generates a plot comparing perecent transmission across different windows on the ear

# # # Input arguments are an XML file, width, and steps
# # # # Output is a scatter plot saved to a folder containing 'XML_filename'.png

import sys
import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#sets parameters for seaborn plots
sns.set_style("darkgrid")


#setting up argparse arguments
parser = argparse.ArgumentParser(description='Given XML file, width, and steps, returns scatterplot')
parser.add_argument('xml', metavar='', help='Input XML filename.', type=str)
parser.add_argument('width', metavar='', help='Width in pixels for the length of the window.', type=int)
parser.add_argument('step_size', metavar='', help='Steps in pixels for window movement.', type=int)
args = parser.parse_args()



def parse_xml(input_xml):
	# Make element tree for object
	tree = ET.parse(input_xml)

	# Getting the root of the tree
	root = tree.getroot()

	# Pulling out the name of the image
	image_name_string = (root[0][0].text)

	# Pulling out the fluorescent and non-fluorescent children
	fluorescent = root[1][1]
	nonfluorescent = root[1][2]

	# Assigning types other than fluorescent and nonfluor in order to
	# # exit program if list is present
	root_4 = root[1][4]
	root_5 = root[1][5]
	root_6 = root[1][6]
	root_7 = root[1][7]
	root_8 = root[1][8]

	if not list(root_4):
		sys.exit(f'Processing {input_xml} stopped...contains unknown type 4.')

	if not list(root_5):
		sys.exit(f'Processing {input_xml} stopped...contains unknown type 5.')

	if not list(root_6):
		sys.exit(f'Processing {input_xml} stopped...contains unknown type 6.')

	if not list(root_7):
		sys.exit(f'Processing {input_xml} stopped...contains unknown type 7.')

	if not list(root_8):
		sys.exit(f'Processing {input_xml} stopped...contains unknown type 8.')


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


def sliding_window(df, w, s):
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
		columns='Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor'.split())

	#Assigning variable to final x coordinate in dataframe
	final_x_coord = df["X-Coordinate"].tail(1)
	int_fxc = int(final_x_coord)

	# # Creating error messages for too small or too big input width or steps
	adj_fxc = int_fxc - 150

	adj_step_fxc = int_fxc * 0.25

	if (int_w >= adj_fxc):
		sys.exit('Width of window is too large, enter smaller width value.')

	if (int_s >= adj_step_fxc):
		sys.exit('Step size is too large, enter smaller value.')

	# # Beginning of sliding window to scan ear and output new dataframe called kern_count_df
	while end_x <= int_fxc:

		#Creating smaller window df based on original df
		rslt_df = df[(df['X-Coordinate'] >= int_start_x) & (df['X-Coordinate'] <= end_x)]

		#Total kernels in window
		kernel_tot = len(rslt_df.index)

		#Error message if there are no kernels in window
		if (kernel_tot == 0):
			sys.exit('0 Kernels in Window, please enter larger width value.')


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
		data = [[steps, window_start, window_end, kernel_tot, fluor_tot, nonfluor_tot]]

		#putting list into dataframe (which is just 1 row)
		data_df = pd.DataFrame(data=data, columns='Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor'.split())

		#appending data_df to kern_count_df (1 row added each time)
		kern_count_df = kern_count_df.append(data_df)

		#shifting window based on stepsize
		int_start_x = int_start_x + steps
		end_x = end_x + steps

	#resetting index
	kern_count_df = kern_count_df.reset_index(drop=True)

	return kern_count_df

#function for plotting total kernels vs average window position
#def tot_kern_scatter ( kern_count_df ):
	#col = kern_count_df.loc[: , "Window_Start":"Window_End"]
	#kern_count_df['window_mean'] = col.mean(axis=1)

	#kern_tot_scatter = sns.scatterplot("window_mean", "Total_Kernels", data=kern_count_df, palette='Set1')
	#tot_kern_figure = kern_tot_scatter.get_figure()
	#tot_kern_figure.savefig("tot_kern_figure.png")
	#my_plot = plt.show()

	#return tot_kern_figure

#function for plotting percent transmission to average window position
# # plots are saved by file name.png and put into new directory called
# # # transmission_plots
def transmission_scatter ( kern_count_df, xml ):
	#calculating average window position
	col = kern_count_df.loc[:, "Window_Start":"Window_End"]
	kern_count_df['window_mean'] = col.mean(axis=1)

	#calculating percent transmission
	kern_count_df['Percent_Transmission'] = kern_count_df['Total_Fluor']/kern_count_df['Total_Kernels']

	kern_count_df.window_mean.astype(float)
	kern_count_df.Percent_Transmission.astype(float)

	#creating plot
	transmission_plot = sns.lineplot(x="window_mean", y="Percent_Transmission", data=kern_count_df)

	sns.set(rc={'figure.figsize': (9, 2.5)})
	plt.gcf().subplots_adjust(bottom=0.3)

	#set title
	plt.title(xml[:-4]+' Plot')
	# Set x-axis label
	plt.xlabel('Window Position (pixels)')
	# Set y-axis label
	plt.ylabel('Percent Transmission')

	transmission_figure = transmission_plot.get_figure()

	#create directory to save plots
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Transmission_plots/')
	#sample_file_name = 'test.png'
	sample_file_name = xml[:-4]+'.png'

	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)

	transmission_figure.savefig(results_dir + sample_file_name)
	plt.close()

	return transmission_figure


#Main function for running the whole script with argparse
# # if - allows you to input xml file as first argument
# # # else - allows you to input directory of xml files as argument
def main():
	if args.xml.endswith(".xml"):
		dataframe = parse_xml(args.xml)
		dataframe2 = sliding_window(dataframe, args.width, args.step_size)
		final_plot = transmission_scatter(dataframe2, args.xml)
	else:
		for roots, dirs, files in os.walk(args.xml):
			for filename in files:
				fullpath = os.path.join(args.xml, filename)
				print(f'Processing {fullpath}')
				if fullpath.endswith(".xml"):
					with open(fullpath, 'r') as f:
						dataframe = parse_xml(f)
						dataframe2 = sliding_window(dataframe, args.width, args.step_size)
						final_plot = transmission_scatter(dataframe2, filename)

	print('Process Complete!')

if __name__ == '__main__':
	main()

