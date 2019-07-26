#!/usr/local/bin/python3

import libpysal as ps
import numpy as np
from pointpats import PointPattern, as_window, G, F, J, K, L, Genv, Fenv, Jenv, Kenv, Lenv
from pointpats import PoissonPointProcess as csr
import matplotlib.pyplot as plt
import pointpats.quadrat_statistics as qs


import sys
import os
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
from pandas import Series, DataFrame
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from pylab import rcParams
from scipy import stats
import pylab as pl
from matplotlib import collections  as mc


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

def spatial_stat(df):
	mtrx = df.as_matrix(columns=df.columns[2:])
	pp_mtrx = PointPattern(mtrx)
	print(pp_mtrx.summary())
	pp_mtrx.plot(window=True, title="Point pattern")
	plt.show()
	q_r = qs.QStatistic(pp_mtrx, shape="rectangle", nx=12, ny=12)
	q_r.plot()
	plt.show()
	print(q_r.chi2) # chisquare stat
	print(q_r.df) # degrees of freedom
	print(q_r.chi2_pvalue) # pvalue

	csr_process = csr(pp_mtrx.window, pp_mtrx.n, 400, asPP=True)
	q_r_e = qs.QStatistic(pp_mtrx, shape="rectangle", nx=12, ny=12, realizations=csr_process)
	x = q_r_e.chi2_r_pvalue
	print(x)
	# print(pp_mtrx.max_nnd) # max nearest neighbor distance
	# print(pp_mtrx.min_nnd) # min
	# print(pp_mtrx.mean_nnd) #mean

	# gp1 = G(pp_mtrx, intervals=20)
	# gp1.plot()
	# plt.show()

	return df


coordinates = parse_xml("/Users/elysevischulis/Downloads/X401x492-2m1.xml")
stat = spatial_stat(coordinates)