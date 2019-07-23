#!/usr/local/bin/python3

#This script creates family group plots from the saved dataframe in txt file

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


#setting up argparse arguments
parser = argparse.ArgumentParser(description='Given meta df, start and stop values for male families, returns plots')
parser.add_argument('input_df', metavar='', help='Input meta dataframe filename.', type=str)
parser.add_argument('start_value', metavar='', help='Starting number for male family plots', type=int)
parser.add_argument('end_value', metavar='', help='Ending number for male family plots', type=int)
args = parser.parse_args()



def everything_everything_graph ( input_df ):
	print(f'Starting everything plot...')

	sns.set_style("white")

	data = pd.read_csv(input_df, sep="\t")


	everything_plot = sns.regplot(x=data["window_mean"], y=data["Percent_Transmission"], fit_reg=True, scatter_kws={"color":"darkred","alpha":0.006,"s":20}, line_kws={"color": "black"})
	sns.set(rc={'figure.figsize': (11.7, 8.27)})
	plt.ylim(0, 1)
	everything_plot.yaxis.grid(True)
	everything_plot.set(yticks=[0, 0.25, 0.5, 0.75, 1])

	plt.title('Everything Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"
	everything_ev_plot = everything_plot.get_figure()
	everything_ev_plot.savefig('everything_plot.png', bbox_inches="tight")

	reg_x = data['window_mean'].values
	reg_y = data['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	print(f'slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')

	print(f'Done everything plot!')
	plt.close()

	return everything_ev_plot


def only400s_plot ( input_df ):
	print(f'Starting everything 400s plot...')

	sns.set_style("white")
	data = pd.read_csv(input_df, sep="\t")

	data = data[data['File'].str.contains('X4')]
	data = data[data['File'].str.contains('x4')]


	justfours = sns.regplot(x=data["window_mean"], y=data["Percent_Transmission"], fit_reg=True, scatter_kws={"color": "darkgreen", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
	sns.set(rc={'figure.figsize': (11.7, 8.27)})
	plt.ylim(0, 1)
	justfours.yaxis.grid(True)
	justfours.set(yticks=[0, 0.25, 0.5, 0.75, 1])

	plt.title('Everything 400s', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"
	justfours_plot = justfours.get_figure()
	justfours_plot.savefig('Everything_400.png', bbox_inches="tight")

	reg_x = data['window_mean'].values
	reg_y = data['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	print(f'slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
	print(f'Done 400s plot!')
	plt.close()

	return justfours_plot


def female_cross_plot (input_df):
	print(f'Starting female plot...')

	sns.set_style("white")

	data = pd.read_csv(input_df, sep="\t")
	data = data[data['File'].str.contains('x4')]

	female = sns.regplot(x=data["window_mean"], y=data["Percent_Transmission"], fit_reg=True,
							scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
	sns.set(rc={'figure.figsize': (11.7, 8.27)})
	plt.ylim(0, 1)
	female.yaxis.grid(True)
	female.set(yticks=[0, 0.25, 0.5, 0.75, 1])

	plt.title('Female 400s Cross Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"
	female_plot = female.get_figure()
	female_plot.savefig('Female 400s Cross Plot.png', bbox_inches="tight")

	reg_x = data['window_mean'].values
	reg_y = data['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	print(f'slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
	print(f'Done female plot!')
	plt.close()

	return female_plot

def male_fam_plot (input_df, low, high):

	print(f'Starting male plots...')
	for i in range(low, high):

		data = pd.read_csv(input_df, sep="\t")
		data = data[data['File'].str.contains(r'x4..')]
		search_values = ['x'+ str(i)]
		data = data[data.File.str.contains('|'.join(search_values))]


		if data.empty:
			continue
		else:
			data['Normalized_Window_Mean'] = data.groupby('File')['window_mean'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

			sns.set_style("white")
			male = sns.lineplot(x="Normalized_Window_Mean", y="Percent_Transmission", data=data, hue="File", linewidth=5)
			sns.set(rc={'figure.figsize': (11.7, 8.27)})
			plt.ylim(0, 1)
			male.set(yticks=[0, 0.25, 0.5, 0.75, 1])
			male.yaxis.grid(True)
			male.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
			plt.title(repr(i) + ' Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
			plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
			plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

			reg_x = data['Normalized_Window_Mean'].values
			reg_y = data['Percent_Transmission'].values
			slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

			print(f'slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
			plt.plot(reg_x, intercept + slope * reg_x, 'r', label='fitted line', color='black', linewidth=3,
					 dashes=[5, 3])

			# def regress (data, x, y):
			# 	r_x = data['x']
			# 	r_y = data['y']
			# 	slope, intercept, r_value, p_value, std_err = stats.linregress(r_x, r_y)
			# 	plt.plot(r_x, intercept + slope * r_x, 'r', label='fitted line', color='red', linewidth=2,
			# 			 dashes=[5, 3])
			# 	return male_graph
			#
			# data.groupby('File').apply[regress(data, data['Normalized_Window_Mean'].values, data['Percent_Transmission'].values)]
			#

			male_graph = male.get_figure()

			# create directory to save plots
			script_dir = os.path.dirname(__file__)
			results_dir = os.path.join(script_dir, 'Male_Plots/')
			# sample_file_name
			sample_file_name = repr(i) + '.png'

			if not os.path.isdir(results_dir):
				os.makedirs(results_dir)

			male_graph.savefig(results_dir + sample_file_name, bbox_inches="tight")
			plt.close()
			print(f'{i} Plot Completed.')
	print(f'Done male plots!')

	return male_graph


def main():
	# everything_everything_plot = everything_everything_graph(args.input_df)
	# fourhundred_plot = only400s_plot(args.input_df)
	# female_plot = female_cross_plot(args.input_df)
	male_fam_graphs = male_fam_plot(args.input_df, args.start_value, args.end_value)


if __name__ == '__main__':
		main()

