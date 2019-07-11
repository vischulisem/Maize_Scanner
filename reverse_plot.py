
import sys
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns




sns.set(rc={'figure.figsize': (9, 2.5)})


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

# Generating plot of coordinate values

#def make_scatter(df):
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
	df.sort_values(by=['X-Coordinate'], inplace=True, ascending=False)
	df = df.reset_index(drop=True)

	# #Choosing starting point for window with value for x
	start_x = df["X-Coordinate"].head(1)
	int_start_x = int(start_x)

	# # setting up w and s as integers
	int_w = int(w)
	int_s = int(s)

	# #Defining end of window
	end_x = int_start_x - int_w

	# #Defining steps for window
	steps = int_s

	# # creating empty dataframe
	kern_count_df = pd.DataFrame(
		columns='Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor'.split())

	final_x_coord = df["X-Coordinate"].tail(1)
	int_fxc = int(final_x_coord)

	adj_fxc = int_fxc + 150

	#adj_step_fxc = int_fxc * 0.25

	#if (int_w <= adj_fxc):
		#sys.exit('Width of window is too large, enter smaller width value.')

	#if (int_s >= adj_step_fxc):
		#sys.exit('Step size is too large, enter smaller value.')


	while end_x >= int_fxc:

		rslt_df = df[(df['X-Coordinate'] <= int_start_x) & (df['X-Coordinate'] >= end_x)]

		kernel_tot = int(len(rslt_df.index))


		if (kernel_tot == 0):
			sys.exit('0 Kernels in Window, please enter larger width value.')

		window_start = int_start_x
		window_end = end_x

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

		data = [[steps, window_start, window_end, kernel_tot, fluor_tot, nonfluor_tot]]
		data_df = pd.DataFrame(data=data, columns='Step_Size Window_Start Window_End Total_Kernels Total_Fluor Total_NonFluor'.split())
		kern_count_df = kern_count_df.append(data_df)

		int_start_x = int_start_x - steps
		end_x = end_x - steps

	kern_count_df = kern_count_df.reset_index(drop=True)

	return kern_count_df


#def tot_kern_scatter ( kern_count_df ):
	#col = kern_count_df.loc[: , "Window_Start":"Window_End"]
	#kern_count_df['window_mean'] = col.mean(axis=1)

	#kern_tot_scatter = sns.scatterplot("window_mean", "Total_Kernels", data=kern_count_df, palette='Set1')
	#tot_kern_figure = kern_tot_scatter.get_figure()
	#tot_kern_figure.savefig("tot_kern_figure.png")
	#my_plot = plt.show()

	#return tot_kern_figure

def transmission_scatter ( kern_count_df ):
	col = kern_count_df.loc[:, "Window_Start":"Window_End"]
	kern_count_df['window_mean'] = col.mean(axis=1)

	kern_count_df['Percent_Transmission'] = kern_count_df['Total_Fluor']/kern_count_df['Total_Kernels']
	transmission_plot = sns.scatterplot("window_mean", "Percent_Transmission", data=kern_count_df, palette='Set1')

	#set title
	plt.title('Reverse Plot')
	# Set x-axis label
	plt.xlabel('Window Position (pixels)')
	# Set y-axis label
	plt.ylabel('Percent Transmission')

	transmission_figure = transmission_plot.get_figure()
	transmission_figure.savefig("reverse_transmission_figure.png")
	my_plot = plt.show()

	return transmission_figure


coordinates = parse_xml("/Users/elysevischulis/Downloads/X401x492-2m1.xml")

ordered_coord = sliding_window(coordinates, 50, 100)

transmission_scatter(ordered_coord)

