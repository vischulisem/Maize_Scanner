#!/usr/local/bin/python3

#This script creates family group plots from the saved dataframe in txt file

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
import re

def everything_everything_graph ( input_df ):
	sns.set_style("white")

	data = pd.read_csv(input_df, sep="\t")


	everything_plot = sns.regplot(x=data["window_mean"], y=data["Percent_Transmission"], fit_reg=True, scatter_kws={"color":"darkred","alpha":0.006,"s":20})
	sns.set(rc={'figure.figsize': (11.7, 8.27)})
	plt.ylim(0, 1)
	everything_plot.yaxis.grid(True)

	plt.title('Everything Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"
	everything_ev_plot = everything_plot.get_figure()
	everything_ev_plot.savefig('everything_plot.png', bbox_inches="tight")

	plt.show()

	reg_x = data['window_mean'].values
	reg_y = data['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	print(f'slope: {slope} intercept: {intercept}')

	return everything_ev_plot

def g_plot (input_df):
	sns.set_style("white")

	datad = pd.read_csv(input_df, sep="\t")

	g = sns.jointplot(x=datad["window_mean"], y=datad["Percent_Transmission"], kind = "kde", space = 0, color = "g")

	# sns.set(rc={'figure.figsize': (11.7, 8.27)})
	# plt.ylim(0, 1)
	# g.yaxis.grid(True)
	plt.title('Everything Plot 2', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	# plt.rcParams["font.weight"] = "bold"
	# plt.rcParams["axes.labelweight"] = "bold"
	g.savefig('everything_plot_2.png', bbox_inches="tight")

	plt.show()

	return g


def only400s_plot ( input_df ):
	sns.set_style("white")
	data = pd.read_csv(input_df, sep="\t")

	data = data[data['File'].str.contains('X4')]
	data = data[data['File'].str.contains('x4')]


	justfours = sns.regplot(x=data["window_mean"], y=data["Percent_Transmission"], fit_reg=True, scatter_kws={"color": "darkgreen", "alpha": 0.006, "s": 20})
	sns.set(rc={'figure.figsize': (11.7, 8.27)})
	plt.ylim(0, 1)
	justfours.yaxis.grid(True)

	plt.title('Everything 400s', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"
	justfours_plot = justfours.get_figure()
	justfours_plot.savefig('Everything_400.png', bbox_inches="tight")

	plt.show()

	reg_x = data['window_mean'].values
	reg_y = data['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	print(f'slope: {slope} intercept: {intercept}')

	return justfours_plot


def female_cross_plot (input_df):
	sns.set_style("white")

	data = pd.read_csv(input_df, sep="\t")
	data = data[data['File'].str.contains('x4')]

	female = sns.regplot(x=data["window_mean"], y=data["Percent_Transmission"], fit_reg=True,
							scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20})
	sns.set(rc={'figure.figsize': (11.7, 8.27)})
	plt.ylim(0, 1)
	female.yaxis.grid(True)

	plt.title('Female 400s Cross Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"
	female_plot = female.get_figure()
	female_plot.savefig('Female 400s Cross Plot.png', bbox_inches="tight")

	plt.show()

	reg_x = data['window_mean'].values
	reg_y = data['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	print(f'slope: {slope} intercept: {intercept}')

	return female_plot

def male_fam_plot (input_df):
	#i = 400
	#while i > 500:
	sns.set_style("white")
	data = pd.read_csv(input_df, sep="\t")
	data = data[data['File'].str.contains(r'x4..')]
	i = 404
	search_values = ['x'+ str(i)]
	data = data[data.File.str.contains('|'.join(search_values))]

	male = sns.lineplot(x="window_mean", y="Percent_Transmission", data=data, hue="File", linewidth=5)
	sns.set(rc={'figure.figsize': (11.7, 8.27)})
	plt.ylim(0, 1)
	male.yaxis.grid(True)

	plt.title('400 Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	male_graph = male.get_figure()

	# create directory to save plots
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Male_Plots/')
	# sample_file_name
	sample_file_name = 'fourhundred.png'

	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)

	male_graph.savefig(results_dir + sample_file_name, bbox_inches="tight")
	plt.close()

		# i = i + 1



	return male_graph



#thisthing = everything_everything_graph("/Users/elysevischulis/Scripts/everything_df.txt")

#new_plot = g_plot("/Users/elysevischulis/Scripts/everything_df.txt")

#four_df = only400s_plot("/Users/elysevischulis/Scripts/everything_df.txt")

#shemale = female_cross_plot("/Users/elysevischulis/Scripts/everything_df.txt")

male_plots = male_fam_plot("/Users/elysevischulis/Scripts/everything_df.txt")