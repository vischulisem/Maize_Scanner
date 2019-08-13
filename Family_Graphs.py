#!/usr/local/bin/python3

# This script creates family group plots from the saved everything_df.txt
# Creates plot with everything on it
# Creates plot with everything with male and female in 400s
# Creates plot with only females in 400s
# Creates plots saved to new directory of Male family plots with regression lines

import os
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches

# Setting up argparse arguments
parser = argparse.ArgumentParser(description='Given everything df.txt, start and stop values for male families, returns plots')
parser.add_argument('-i', '--input_df', metavar='', help='Input everything dataframe filename.', type=str)
parser.add_argument('-t', '--table8', metavar='', help='Supplemental table 8 for expression and transmission data.', type=str)
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
    # Setting text font to bold
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
    female = sns.regplot(x=data["window_mean"], y=data["Percent_Transmission"], fit_reg=True, scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
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

# Plots everything in .txt file with normalized x axis
def everything_norm_everything_graph(input_df, path):
    print(f'Starting everything plot...')

    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.yaxis.grid(color='gainsboro')
    # Reading in txt file as pandas df
    data = pd.read_csv(input_df, sep="\t")
    # Plotting begins
    sns.regplot(x=data["Normalized_Window_Mean"], y=data["Percent_Transmission"], fit_reg=True, scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plt.ylim(0, 1)
    ax.yaxis.grid(True)
    ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])

    plt.title('Everything Norm Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
    plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
    plt.ylabel('Percent Transmission', fontsize=18, weight='bold')
    # Bold text
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # Drawing black box around plot
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    # Labelling x axis with bottom and top of ear
    fig.canvas.draw()

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[6] = 'Top'
    labels[1] = 'Bottom'
    labels[2] = 0.2
    labels[3] = 0.4
    labels[4] = 0.6
    labels[5] = 0.8

    ax.set_xticklabels(labels)
    # Saving Figure
    everything_ev_plot = plt.gcf()
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

# Plots only xml files in .txt with X4..x4... with normalized x axis
def only400s_norm_plot(input_df, path):
    print(f'Starting everything 400s plot...')

    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.yaxis.grid(color='gainsboro')
    # Reading in file as dataframe
    data = pd.read_csv(input_df, sep="\t")
    # Sorting through file name column
    data = data[data['File'].str.contains('X4')]
    data = data[data['File'].str.contains('x4')]
    # Plotting starts
    sns.regplot(x=data["Normalized_Window_Mean"], y=data["Percent_Transmission"], fit_reg=True, scatter_kws={"color": "darkgreen", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plt.ylim(0, 1)
    ax.yaxis.grid(True)
    ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])

    plt.title('Everything Norm 400s', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
    plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
    plt.ylabel('Percent Transmission', fontsize=18, weight='bold')
    # Bold text
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # Drawing black box around plot
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    # Labeling top and bottom of ear on x axis
    fig.canvas.draw()

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[6] = 'Top'
    labels[1] = 'Bottom'
    labels[2] = 0.2
    labels[3] = 0.4
    labels[4] = 0.6
    labels[5] = 0.8

    ax.set_xticklabels(labels)
    # Saving Figure
    justfours_plot = plt.gcf()
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

# Plots only xml files named ....x4.... with normalized x axis
def female_cross_norm_plot(input_df, path):
    print(f'Starting female plot...')
    fig, ax = plt.subplots()

    # Reading in txt file as dataframe
    data = pd.read_csv(input_df, sep="\t")
    # Sorting through file names
    data = data[data['File'].str.contains('x4')]
    # Plotting begins
    sns.regplot(x=data["Normalized_Window_Mean"], y=data["Percent_Transmission"], fit_reg=True, scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plt.ylim(0, 1)
    ax.yaxis.grid(True)
    ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])

    plt.title('Female 400s Norm Cross Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
    plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
    plt.ylabel('Percent Transmission', fontsize=18, weight='bold')
    ax.set_facecolor('white')
    ax.yaxis.grid(color='gainsboro')
    # Bold text
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # Drawing black box around plot
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    # Labeling top and bottom of ear on x axis
    fig.canvas.draw()

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[6] = 'Top'
    labels[1] = 'Bottom'
    labels[2] = 0.2
    labels[3] = 0.4
    labels[4] = 0.6
    labels[5] = 0.8

    ax.set_xticklabels(labels)
    # Saving Figure
    female_plot = plt.gcf()

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
    big_reg_df = pd.DataFrame(columns='Male_Fam Normalized_Window_Mean Percent_Transmission'.split())
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
            fig, ax = plt.subplots()
            sns.lineplot(x="Normalized_Window_Mean", y="Percent_Transmission", data=data, hue="File", linewidth=5)
            sns.set(rc={'figure.figsize': (11.7, 8.27)})
            plt.ylim(0, 1)
            ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])
            ax.yaxis.grid(True)
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
            plt.title(repr(i) + ' Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
            plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
            plt.ylabel('Percent Transmission', fontsize=18, weight='bold')
            ax.set_facecolor('white')
            ax.yaxis.grid(color='gainsboro')
            # Regression calculations
            reg_x = data['Normalized_Window_Mean'].values
            reg_y = data['Percent_Transmission'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
            r2 = r_value ** 2
            # Putting stats for each family in dataframe for use in later scripts
            reg_xx = data['Normalized_Window_Mean'].values.tolist()
            reg_yy = data['Percent_Transmission'].values.tolist()
            llist = [str(i)] * len(reg_xx)
            reg_list = [list(a) for a in zip(llist, reg_xx, reg_yy)]
            reg_df = pd.DataFrame(data=reg_list, columns='Male_Fam Normalized_Window_Mean Percent_Transmission'.split())
            big_reg_df = big_reg_df.append(reg_df)
            big_reg_df = big_reg_df.reset_index(drop=True)

            # Creating empty df so that mean regression stats for each line can be displayed
            stat_df = pd.DataFrame(columns='Slope Intercept RSquared P-Value'.split())
            # Iterating through each file name and plotting regression line individually
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

            # Plotting regression line for all files in fam together
            plt.plot(reg_x, intercept + slope * reg_x, 'r', label='fitted line', color='white', linewidth=6)
            plt.plot(reg_x, intercept + slope * reg_x, 'r', label='fitted line', color='red', linewidth=4)

            # Calculating mean values for each stat and rounding to 4 decimal places
            average_slope = round(stat_df['Slope'].mean(), 4)
            average_intercept = round(stat_df['Intercept'].mean(), 4)
            average_Rsquared = round(stat_df['RSquared'].mean(), 4)
            average_Pval = '{:0.3e}'.format(stat_df['P-Value'].mean())
            combined_pval = '{:0.3e}'.format(p_value)
            # Text string for text box
            textstr = '\n'.join((f'Combined Slope = {round(slope, 4)}',
                                 f'Average Indv. Slope = {average_slope}',
                                 f'Combined Intercept = {round(intercept, 4)}',
                                 f'Average Indv. Intercept = {average_intercept}',
                                 f'Combined P-value = {combined_pval}',
                                 f'Average Indv. P-value = {average_Pval}',
                                 f'Combined R-Squared = {round(r_value ** 2, 4)}',
                                 f'Average Indv. R-squared = {average_Rsquared}'))
            # Creating text box on graph
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, fontweight='bold',
                    verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10, 'edgecolor': 'black'})
            # Drawing black box around plot
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black')
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            # Labelling top and bottom of ear on x axis
            fig.canvas.draw()

            labels = [item.get_text() for item in ax.get_xticklabels()]
            labels[6] = 'Top'
            labels[1] = 'Bottom'
            labels[2] = 0.2
            labels[3] = 0.4
            labels[4] = 0.6
            labels[5] = 0.8

            ax.set_xticklabels(labels)

            # Saving figure in new directory with cross file name
            male_graph = plt.gcf()

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
    big_reg_df['Male_Fam'] = big_reg_df['Male_Fam'].astype(np.int64)
    return big_reg_df

# Function to plot regression lines from each family colored by cell expression
# Requires supplemental table 8 which was saved to txt file in Data_for_analysis folder
def reg_exp(table8, big_reg_df, path):
    # Reading table into dataframe
    eight_df = pd.read_csv(table8, sep="\t")
    # Removing columns crossed through female
    eight_df = eight_df[eight_df['Mutant allele parent'] != 'female']
    # Selecting columns we're interested in
    eight_df = eight_df.loc[:, ['Tracking number', 'Expression class', 'Adjusted p-value']]
    # Renaming a column for combining
    eight_df = eight_df.rename({'Tracking number': 'Male_Fam'}, axis=1)
    # Converting values to int for combining
    eight_df['Male_Fam'] = eight_df['Male_Fam'].astype(np.int64)
    # Merge two dfs to giant df along 'Male_Fam' column
    merged_df = big_reg_df.merge(eight_df, how='inner', on=['Male_Fam'])

    # Plotting begins of regression line for each male family
    # In loop, groups by male fam and plots line
    fig, ax = plt.subplots()
    grps = merged_df.groupby(['Male_Fam'])
    for file, grp in grps:
        iter_y = grp['Percent_Transmission']
        iter_x = grp['Normalized_Window_Mean']
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(iter_x, iter_y)
        # Coloring based on expression
        if (grp['Expression class'] == 'vegetative_cell_high').any():
            c = 'dimgrey'
            z = 1
        elif (grp['Expression class'] == 'seedling_only').any():
            c = 'deepskyblue'
            z = 2
        else:
            c = 'orange'
            z = 3
        plt.plot(iter_x, intercept2 + slope2 * iter_x, 'r', label='fitted line', color=c, linewidth=1.5, zorder=z)
    # Graph aesthetics
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plt.ylim(0, 1)
    ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.grid(True)
    plt.title('Expression Class Regression Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
    plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
    plt.ylabel('Percent Transmission', fontsize=18, weight='bold')
    # Creating legend
    black_patch = mpatches.Patch(color='dimgrey', label='Vegetative Cell')
    aqua_patch = mpatches.Patch(color='deepskyblue', label='Seedling')
    orange_patch = mpatches.Patch(color='orange', label='Sperm Cell')
    ax.legend(handles=[black_patch, aqua_patch, orange_patch], loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_facecolor('white')
    ax.yaxis.grid(color='gainsboro')
    # Drawing black box around plot
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    # labelling top and bottom of ear
    fig.canvas.draw()

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[6] = 'Top'
    labels[1] = 'Bottom'
    labels[2] = 0.2
    labels[3] = 0.4
    labels[4] = 0.6
    labels[5] = 0.8

    ax.set_xticklabels(labels)

    # Saving figure in new directory with cross file name
    reg_expgraph = plt.gcf()

    # create directory to save plots
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Family_Graphs/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # sample_file_name
    sample_file_name = 'Exp_Reg_Plot.png'

    reg_expgraph.savefig(results_dir + sample_file_name, bbox_inches="tight")
    plt.close()
    return merged_df

# Plot male fam regression lines colored based on whether known transmission defect or not
# Similar to previous function
def reg_trans(merged_df, path):
    # Plotting begins, cycle through each family and plot regression line
    fig, ax = plt.subplots()
    grps = merged_df.groupby(['Male_Fam'])
    for file, grp in grps:
        iter_y = grp['Percent_Transmission']
        iter_x = grp['Normalized_Window_Mean']
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(iter_x, iter_y)
        # Color based on adjusted p value
        if (grp['Adjusted p-value'] > 0.05).any():
            c = 'black'
            z = 1
        else:
            c = 'red'
            z = 2
        plt.plot(iter_x, intercept2 + slope2 * iter_x, 'r', label='fitted line', color=c, linewidth=1.5, zorder=z)

    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plt.ylim(0, 1)
    ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.grid(True)
    plt.title('Transmission Defect Regression Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
    plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
    plt.ylabel('Percent Transmission', fontsize=18, weight='bold')
    # Making legend
    black_patch = mpatches.Patch(color='black', label='No defect')
    red_patch = mpatches.Patch(color='red', label='Transmission defect')
    ax.legend(handles=[black_patch, red_patch], loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_facecolor('white')
    ax.yaxis.grid(color='gainsboro')
    # Drawing black box around graph
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    # Labelling top and bottom of ear on x axis
    fig.canvas.draw()

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[6] = 'Top'
    labels[1] = 'Bottom'
    labels[2] = 0.2
    labels[3] = 0.4
    labels[4] = 0.6
    labels[5] = 0.8

    ax.set_xticklabels(labels)

    # Saving figure in new directory with cross file name
    reg_transgraph = plt.gcf()

    # create directory to save plots
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Family_Graphs/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # sample_file_name
    sample_file_name = 'Trans_Reg_Plot.png'

    reg_transgraph.savefig(results_dir + sample_file_name, bbox_inches="tight")
    plt.close()
    return merged_df

# Main function to run everything
def main():
    # If you want x axis normalized
    if args.n:
        everything_norm_everything_graph(args.input_df, args.path)
        only400s_norm_plot(args.input_df, args.path)
        female_cross_norm_plot(args.input_df, args.path)
    # X axis with original pixels
    else:
        everything_everything_graph(args.input_df, args.path)
        only400s_plot(args.input_df, args.path)
        female_cross_plot(args.input_df, args.path)
    big_reg_df = male_fam_plot(args.input_df, args.start_value, args.end_value, args.path)
    merged_df = reg_exp(args.table8, big_reg_df, args.path)
    reg_trans(merged_df, args.path)

if __name__ == '__main__':
    main()