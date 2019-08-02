#!/usr/local/bin/python3

# This script creates family group plots from the saved everything_df.txt
# Creates plot with everything on it
# Creates plot with everything with male and female in 400s
# Creates plot with only females in 400s
# Creates plots saved to new directory of Male family plots with regression lines

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# setting up argparse arguments
parser = argparse.ArgumentParser(description='Given everything df, start and stop values for male families, returns plots')
parser.add_argument('-i', '--input_df', metavar='', help='Input meta dataframe filename.', type=str)
parser.add_argument('-sv', '--start_value', metavar='', help='Starting number for male family plots, must be in 400s', default=411, type=int)
parser.add_argument('-ev', '--end_value', metavar='', help='Ending number for male family plots, must be in 400s', default=499, type=int)
parser.add_argument('-n', action='store_true', help='Will normalize x axis of transmission plots.')
parser.add_argument('-p', '--path', metavar='', help='List path where you want files saved to.', default=os.getcwd(), type=str)
args = parser.parse_args()

# Plots everything in .txt file
def everything_everything_graph(input_df, path):
	print(f'Starting everything plot...')

	sns.set_style("white")
	# Reading in txt file as pandas df
	data = pd.read_csv(input_df, sep="\t")
	# Plotting begins
	everything_plot = sns.regplot(x=data["window_mean"], y=data["Percent_Transmission"], fit_reg=True, scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
	sns.set(rc={'figure.figsize': (11.7, 8.27)})
	plt.ylim(0, 1)
	everything_plot.yaxis.grid(True)
	everything_plot.set(yticks=[0, 0.25, 0.5, 0.75, 1])

	plt.title('Everything Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"
	# Saving figure
	everything_ev_plot = everything_plot.get_figure()
	script_dir = path
	results_dir = os.path.join(script_dir, 'Output_Family_Graphs/')
	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)
	everything_ev_plot.savefig(results_dir + 'everything_plot.png', bbox_inches="tight")
	# Calculating regression
	reg_x = data['window_mean'].values
	reg_y = data['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	print(f'slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')

	print(f'Done everything plot!')
	plt.close()

	return everything_ev_plot

# Plots only xml files in .txt with X4..x4...
def only400s_plot(input_df, path):
	print(f'Starting everything 400s plot...')

	sns.set_style("white")
	# Reading in file as dataframe
	data = pd.read_csv(input_df, sep="\t")
	# Sorting through file name column
	data = data[data['File'].str.contains('X4')]
	data = data[data['File'].str.contains('x4')]
	# Plotting starts
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
	# Saving figure
	justfours_plot = justfours.get_figure()
	script_dir = path
	results_dir = os.path.join(script_dir, 'Output_Family_Graphs/')
	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)
	justfours_plot.savefig(results_dir + 'Everything_400.png', bbox_inches="tight")
	# Regression Calculation
	reg_x = data['window_mean'].values
	reg_y = data['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	print(f'slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
	print(f'Done 400s plot!')
	plt.close()

	return justfours_plot

# Plots only xml files named ....x4....
def female_cross_plot(input_df, path):
	print(f'Starting female plot...')

	sns.set_style("white")
	# Reading in txt file as dataframe
	data = pd.read_csv(input_df, sep="\t")
	# Sorting through file names
	data = data[data['File'].str.contains('x4')]
	# Plotting begins
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
	# Saving Figure
	female_plot = female.get_figure()
	script_dir = path
	results_dir = os.path.join(script_dir, 'Output_Family_Graphs/')
	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)
	female_plot.savefig(results_dir + 'Female 400s Cross Plot.png', bbox_inches="tight")
	# Calculating regression
	reg_x = data['window_mean'].values
	reg_y = data['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	print(f'slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
	print(f'Done female plot!')
	plt.close()

	return female_plot

# Plots everything in .txt file
def everything_norm_everything_graph(input_df, path):
	print(f'Starting everything plot...')

	sns.set_style("white")
	# Reading in txt file as pandas df
	data = pd.read_csv(input_df, sep="\t")
	# Plotting begins
	everything_plot = sns.regplot(x=data["Normalized_Window_Mean"], y=data["Percent_Transmission"], fit_reg=True, scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
	sns.set(rc={'figure.figsize': (11.7, 8.27)})
	plt.ylim(0, 1)
	everything_plot.yaxis.grid(True)
	everything_plot.set(yticks=[0, 0.25, 0.5, 0.75, 1])

	plt.title('Everything Norm Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"
	# Saving figure
	everything_ev_plot = everything_plot.get_figure()
	script_dir = path
	results_dir = os.path.join(script_dir, 'Output_Family_Graphs/')
	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)
	everything_ev_plot.savefig(results_dir + 'everything_norm_plot.png', bbox_inches="tight")
	# Calculating regression
	reg_x = data['Normalized_Window_Mean'].values
	reg_y = data['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	print(f'slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')

	print(f'Done everything plot!')
	plt.close()

	return everything_ev_plot

# Plots only xml files in .txt with X4..x4...
def only400s_norm_plot(input_df, path):
	print(f'Starting everything 400s plot...')

	sns.set_style("white")
	# Reading in file as dataframe
	data = pd.read_csv(input_df, sep="\t")
	# Sorting through file name column
	data = data[data['File'].str.contains('X4')]
	data = data[data['File'].str.contains('x4')]
	# Plotting starts
	justfours = sns.regplot(x=data["Normalized_Window_Mean"], y=data["Percent_Transmission"], fit_reg=True, scatter_kws={"color": "darkgreen", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
	sns.set(rc={'figure.figsize': (11.7, 8.27)})
	plt.ylim(0, 1)
	justfours.yaxis.grid(True)
	justfours.set(yticks=[0, 0.25, 0.5, 0.75, 1])

	plt.title('Everything Norm 400s', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"
	# Saving figure
	justfours_plot = justfours.get_figure()
	script_dir = path
	results_dir = os.path.join(script_dir, 'Output_Family_Graphs/')
	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)
	justfours_plot.savefig(results_dir + 'Everything_norm_400.png', bbox_inches="tight")
	# Regression Calculation
	reg_x = data['Normalized_Window_Mean'].values
	reg_y = data['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	print(f'slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
	print(f'Done 400s plot!')
	plt.close()

	return justfours_plot

# Plots only xml files named ....x4....
def female_cross_norm_plot(input_df, path):
	print(f'Starting female plot...')

	sns.set_style("white")
	# Reading in txt file as dataframe
	data = pd.read_csv(input_df, sep="\t")
	# Sorting through file names
	data = data[data['File'].str.contains('x4')]
	# Plotting begins
	female = sns.regplot(x=data["Normalized_Window_Mean"], y=data["Percent_Transmission"], fit_reg=True,
							scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
	sns.set(rc={'figure.figsize': (11.7, 8.27)})
	plt.ylim(0, 1)
	female.yaxis.grid(True)
	female.set(yticks=[0, 0.25, 0.5, 0.75, 1])

	plt.title('Female 400s Norm Cross Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
	plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
	plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"
	# Saving Figure
	female_plot = female.get_figure()
	script_dir = path
	results_dir = os.path.join(script_dir, 'Output_Family_Graphs/')
	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)
	female_plot.savefig(results_dir + 'Female 400s Norm Cross Plot.png', bbox_inches="tight")
	# Calculating regression
	reg_x = data['Normalized_Window_Mean'].values
	reg_y = data['Percent_Transmission'].values
	slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

	print(f'slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
	print(f'Done female plot!')
	plt.close()

	return female_plot

# User can input low and high parameter corresponding to number of 400 male family to plot
# Files are sorted and grouped together, x coordinates (window mean) normalized for all lines
def male_fam_plot(input_df, low, high, path):
	print(f'Starting male plots...')
	for i in range(low, high):
		# Reading in txt file as dataframe
		data = pd.read_csv(input_df, sep="\t")
		# Sorting through file names while iterating through range
		data = data[data['File'].str.contains(r'x4..')]
		search_values = ['x' + str(i)]
		data = data[data.File.str.contains('|'.join(search_values))]

		# If family doesn't exist as filename, go to next i
		if data.empty:
			continue
		else:
			# Group all filenames together and normalize window mean x coordinates
			data['Normalized_Window_Mean'] = data.groupby('File')['window_mean'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
			# Plotting begins
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
			# Regression calculations
			reg_x = data['Normalized_Window_Mean'].values
			reg_y = data['Percent_Transmission'].values
			slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)

			print(f'slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
			# Plotting regression line
			plt.plot(reg_x, intercept + slope * reg_x, 'r', label='fitted line', color='red', linewidth=3,
					 dashes=[5, 3])

			# Creating empty df so that mean rsquared for each line can be displayed
			stat_df = pd.DataFrame(columns='Slope Intercept RSquared P-Value'.split())
			# Iterating through each file name and plotting regression line
			grps = data.groupby(['File'])
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

			# Text string for text box
			textstr = '\n'.join((f'Average R-squared = {average_Rsquared}',
								 f'Average Slope = {average_slope}',
								 f'Average Intercept = {average_intercept}',
								 f'Average P-value = {average_Pval}'))
			# Creating text box on graph
			male.text(0.05, 0.95, textstr, transform=male.transAxes, fontsize=10, fontweight='bold',
					verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10, 'edgecolor': 'black'})

			# Saving figure in new directory with cross file name
			male_graph = male.get_figure()

			# create directory to save plots
			script_dir = path
			results_dir = os.path.join(script_dir, 'Output_Family_Graphs/Male_Plots/')
			if not os.path.isdir(results_dir):
				os.makedirs(results_dir)
			# sample_file_name
			sample_file_name = repr(i) + '.png'

			male_graph.savefig(results_dir + sample_file_name, bbox_inches="tight")
			plt.close()
			print(f'{i} Plot Completed.')
	print(f'Done male plots!')

	return male_graph

def main():
	if args.n:
		everything_norm_everything_graph(args.input_df, args.path)
		only400s_norm_plot(args.input_df, args.path)
		female_cross_norm_plot(args.input_df, args.path)
	else:
		everything_everything_graph(args.input_df, args.path)
		only400s_plot(args.input_df, args.path)
		female_cross_plot(args.input_df, args.path)
	male_fam_plot(args.input_df, args.start_value, args.end_value, args.path)

if __name__ == '__main__':
	main()