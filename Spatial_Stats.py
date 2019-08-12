#!/usr/local/bin/python3

# This is a rough script on testing some spatial stats using the coordinates from xml files

from pointpats import PointPattern, as_window, G, F, J, K, L, Genv, Fenv, Jenv, Kenv, Lenv
from pointpats import PoissonPointProcess as csr
import pointpats.quadrat_statistics as qs
import os
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

# Setting up argparse arguments
parser = argparse.ArgumentParser(description='Given XML file, width, and steps, returns scatterplot')
parser.add_argument('-x', '--xml', metavar='', help='Input XML filename or directory.', type=str)
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

	return df, image_name_string


def spatial_stat(df, image_name_string):
	mtrx = df.as_matrix(columns=df.columns[2:])
	pp_mtrx = PointPattern(mtrx)
	# print(pp_mtrx.summary())
	# pp_mtrx.plot(window=True, title="Point pattern")
	# plt.show()
	q_r = qs.QStatistic(pp_mtrx, shape="rectangle", nx=3, ny=3)
	# q_r.plot()
	# plt.show()
	# print(f'Chi squared: {q_r.chi2}') # chisquare stat
	# print(f'Degrees of freedom: {q_r.df}') # degrees of freedom
	# print(f'P-value: {q_r.chi2_pvalue}') # pvalue
	# print(f'If P value is less than 0.05 then reject null at 95% confidence level.')

	# Empirical sampling distribution
	csr_process = csr(pp_mtrx.window, pp_mtrx.n, 999, asPP=True)
	q_r_e = qs.QStatistic(pp_mtrx, shape="rectangle", nx=3, ny=3, realizations=csr_process)
	x = q_r_e.chi2_r_pvalue
	# print(f'The pseudo p-value...')
	# print(x)
	# print(f'If P value is less than 0.05 then reject null at 95% confidence level.')

	temp_list = [[image_name_string, q_r.chi2, q_r.df, q_r.chi2_pvalue, x]]
	stat_df = pd.DataFrame(data=temp_list, columns='File Chi_Squared DoF P-Value Pseudo_P-Value'.split())

	return stat_df

def main():
	# Dataframe (saved to .txt file) for each file and their chisquare and regression stats
	total_stat_df = pd.DataFrame(columns='File Chi_Squared DoF P-Value Pseudo_P-Value'.split())

	for roots, dirs, files in os.walk(args.xml):
		for filename in files:
			fullpath = os.path.join(args.xml, filename)
			print(f'Processing {fullpath}...')
			if fullpath.endswith(".xml"):
				with open(fullpath, 'r') as f:
					result, tree, count_7, count_8 = check_xml_error(f)
					if result == 'True':
						continue
					dataframe, name = parse_xml(f, tree, count_7, count_8)
					dataframe2 = spatial_stat(dataframe, name)

					total_stat_df = total_stat_df.append(dataframe2)
					total_stat_df = total_stat_df.reset_index(drop=True)

	# Saving dataframes
	total_stat_df = total_stat_df.sort_values(by='P-Value', ascending=False)
	total_stat_df = total_stat_df.reset_index(drop=True)
	total_stat_df.to_csv('Total_Spatial_Stat_df.txt', sep='\t')
	print('Process Complete!')

if __name__ == '__main__':
	main()