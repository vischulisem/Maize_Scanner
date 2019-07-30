#!/usr/local/bin/python3

# This script creates MODEL group plots from the saved everything_model_df.txt
# Creates MODEL plot with everything on it
# Creates MODEL plot with everything with male and female in 400s
# Creates MODEL plot with only females in 400s
# 5 models are layered on top of eachother in each graph and regression lines calculated

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# setting up argparse arguments
parser = argparse.ArgumentParser(description='Given meta df, start and stop values for male families, returns plots')
parser.add_argument('-i', '--input_df', metavar='', help='Input meta dataframe filename.', type=str)
parser.add_argument('-sv', '--start_value', metavar='', help='Starting number for male family plots', type=int)
parser.add_argument('-ev', '--end_value', metavar='', help='Ending number for male family plots', type=int)
parser.add_argument('-n', action='store_true', help='Will normalize x axis of transmission plots.')
args = parser.parse_args()


# Plots everything in everything_model_df.txt file
def everything_everything_graph(input_df):
    print(f'Starting everything plot...')

    sns.set_style("white")
    # Reading in txt file as pandas df
    data = pd.read_csv(input_df, sep="\t")
    # Plotting begins
    fig, ax = plt.subplots()
    sns.regplot(x="window_mean", y="Percent_Transmission", ax=ax, data=data, fit_reg=True, scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="window_mean", y="Percent_Transmission2", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="window_mean", y="Percent_Transmission3", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="window_mean", y="Percent_Transmission4", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="window_mean", y="Percent_Transmission5", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})

    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plt.ylim(0, 1)
    ax.yaxis.grid(True)
    ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])

    plt.title('Everything Model Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
    plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
    plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # Saving figure
    fig.get_figure()
    fig.savefig('everything_MODEL_plot.png', bbox_inches="tight")
    # Calculating regression for each model
    reg_x = data['window_mean'].values
    reg2_y = data['Percent_Transmission2'].values
    reg3_y = data['Percent_Transmission3'].values
    reg4_y = data['Percent_Transmission4'].values
    reg5_y = data['Percent_Transmission5'].values
    reg_y = data['Percent_Transmission'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(reg_x, reg2_y)
    slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(reg_x, reg3_y)
    slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(reg_x, reg4_y)
    slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(reg_x, reg5_y)
    print(f'Model 1: slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
    print(f'Model 2: slope: {slope2} intercept: {intercept2} p-value: {p_value2} R-squared: {r_value2 ** 2}')
    print(f'Model 3: slope: {slope3} intercept: {intercept3} p-value: {p_value3} R-squared: {r_value3 ** 2}')
    print(f'Model 4: slope: {slope4} intercept: {intercept4} p-value: {p_value4} R-squared: {r_value4 ** 2}')
    print(f'Model 5: slope: {slope5} intercept: {intercept5} p-value: {p_value5} R-squared: {r_value5 ** 2}')
    print(f'Done everything plot!')
    plt.close()

    return fig


# Plots only xml files in everything_model_df.txt with X4..x4...
def only400s_plot(input_df):
    print(f'Starting everything 400s plot...')

    sns.set_style("white")
    # Reading in file as dataframe
    data = pd.read_csv(input_df, sep="\t")
    # Sorting through file name column
    data = data[data['File'].str.contains('X4')]
    data = data[data['File'].str.contains('x4')]
    # Plotting starts
    fig, ax = plt.subplots()
    sns.regplot(x="window_mean", y="Percent_Transmission", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "green", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="window_mean", y="Percent_Transmission2", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "green", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="window_mean", y="Percent_Transmission3", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "green", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="window_mean", y="Percent_Transmission4", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "green", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="window_mean", y="Percent_Transmission5", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "green", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plt.ylim(0, 1)
    ax.yaxis.grid(True)
    ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])

    plt.title('Everything MODEL 400s', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
    plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
    plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # Saving figure
    fig.get_figure()
    fig.savefig('Everything_400_MODEL.png', bbox_inches="tight")
    # Regression Calculation for each model
    reg_x = data['window_mean'].values
    reg2_y = data['Percent_Transmission2'].values
    reg3_y = data['Percent_Transmission3'].values
    reg4_y = data['Percent_Transmission4'].values
    reg5_y = data['Percent_Transmission5'].values
    reg_y = data['Percent_Transmission'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(reg_x, reg2_y)
    slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(reg_x, reg3_y)
    slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(reg_x, reg4_y)
    slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(reg_x, reg5_y)
    print(f'Model 1: slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
    print(f'Model 2: slope: {slope2} intercept: {intercept2} p-value: {p_value2} R-squared: {r_value2 ** 2}')
    print(f'Model 3: slope: {slope3} intercept: {intercept3} p-value: {p_value3} R-squared: {r_value3 ** 2}')
    print(f'Model 4: slope: {slope4} intercept: {intercept4} p-value: {p_value4} R-squared: {r_value4 ** 2}')
    print(f'Model 5: slope: {slope5} intercept: {intercept5} p-value: {p_value5} R-squared: {r_value5 ** 2}')

    print(f'Done 400s plot!')
    plt.close()

    return fig


# Plots only xml files named ....x4....
def female_cross_plot(input_df):
    print(f'Starting female plot...')

    sns.set_style("white")
    # Reading in txt file as dataframe
    data = pd.read_csv(input_df, sep="\t")
    # Sorting through file names
    data = data[data['File'].str.contains('x4')]
    # Plotting begins
    fig, ax = plt.subplots()
    sns.regplot(x="window_mean", y="Percent_Transmission", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="window_mean", y="Percent_Transmission2", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="window_mean", y="Percent_Transmission3", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="window_mean", y="Percent_Transmission4", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="window_mean", y="Percent_Transmission5", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})

    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plt.ylim(0, 1)
    ax.yaxis.grid(True)
    ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])

    plt.title('Female MODEL 400s Cross Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
    plt.xlabel('Window Position (pixels)', fontsize=18, weight='bold')
    plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # Saving Figure
    fig.get_figure()
    fig.savefig('Female MODEL 400s Cross Plot.png', bbox_inches="tight")
    # Calculating regression for each model
    reg_x = data['window_mean'].values
    reg2_y = data['Percent_Transmission2'].values
    reg3_y = data['Percent_Transmission3'].values
    reg4_y = data['Percent_Transmission4'].values
    reg5_y = data['Percent_Transmission5'].values
    reg_y = data['Percent_Transmission'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(reg_x, reg2_y)
    slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(reg_x, reg3_y)
    slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(reg_x, reg4_y)
    slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(reg_x, reg5_y)
    print(f'Model 1: slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
    print(f'Model 2: slope: {slope2} intercept: {intercept2} p-value: {p_value2} R-squared: {r_value2 ** 2}')
    print(f'Model 3: slope: {slope3} intercept: {intercept3} p-value: {p_value3} R-squared: {r_value3 ** 2}')
    print(f'Model 4: slope: {slope4} intercept: {intercept4} p-value: {p_value4} R-squared: {r_value4 ** 2}')
    print(f'Model 5: slope: {slope5} intercept: {intercept5} p-value: {p_value5} R-squared: {r_value5 ** 2}')

    print(f'Done female plot!')
    plt.close()

    return fig

# Plots everything in everything_model_df.txt file
def everything_norm_everything_graph(input_df):
    print(f'Starting everything plot...')

    sns.set_style("white")
    # Reading in txt file as pandas df
    data = pd.read_csv(input_df, sep="\t")
    # Plotting begins
    fig, ax = plt.subplots()
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission", ax=ax, data=data, fit_reg=True, scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission2", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission3", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission4", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission5", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkred", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})

    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plt.ylim(0, 1)
    ax.yaxis.grid(True)
    ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])

    plt.title('Everything Model Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
    plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
    plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # Saving figure
    fig.get_figure()
    fig.savefig('everything_norm_MODEL_plot.png', bbox_inches="tight")
    # Calculating regression for each model
    reg_x = data['Normalized_Window_Mean'].values
    reg2_y = data['Percent_Transmission2'].values
    reg3_y = data['Percent_Transmission3'].values
    reg4_y = data['Percent_Transmission4'].values
    reg5_y = data['Percent_Transmission5'].values
    reg_y = data['Percent_Transmission'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(reg_x, reg2_y)
    slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(reg_x, reg3_y)
    slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(reg_x, reg4_y)
    slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(reg_x, reg5_y)
    print(f'Model 1: slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
    print(f'Model 2: slope: {slope2} intercept: {intercept2} p-value: {p_value2} R-squared: {r_value2 ** 2}')
    print(f'Model 3: slope: {slope3} intercept: {intercept3} p-value: {p_value3} R-squared: {r_value3 ** 2}')
    print(f'Model 4: slope: {slope4} intercept: {intercept4} p-value: {p_value4} R-squared: {r_value4 ** 2}')
    print(f'Model 5: slope: {slope5} intercept: {intercept5} p-value: {p_value5} R-squared: {r_value5 ** 2}')
    print(f'Done everything plot!')
    plt.close()

    return fig


# Plots only xml files in everything_model_df.txt with X4..x4...
def only400s_norm_plot(input_df):
    print(f'Starting everything 400s plot...')

    sns.set_style("white")
    # Reading in file as dataframe
    data = pd.read_csv(input_df, sep="\t")
    # Sorting through file name column
    data = data[data['File'].str.contains('X4')]
    data = data[data['File'].str.contains('x4')]
    # Plotting starts
    fig, ax = plt.subplots()
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "green", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission2", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "green", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission3", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "green", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission4", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "green", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission5", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "green", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plt.ylim(0, 1)
    ax.yaxis.grid(True)
    ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])

    plt.title('Everything MODEL 400s', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
    plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
    plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # Saving figure
    fig.get_figure()
    fig.savefig('Everything_400_norm_MODEL.png', bbox_inches="tight")
    # Regression Calculation for each model
    reg_x = data['Normalized_Window_Mean'].values
    reg2_y = data['Percent_Transmission2'].values
    reg3_y = data['Percent_Transmission3'].values
    reg4_y = data['Percent_Transmission4'].values
    reg5_y = data['Percent_Transmission5'].values
    reg_y = data['Percent_Transmission'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(reg_x, reg2_y)
    slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(reg_x, reg3_y)
    slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(reg_x, reg4_y)
    slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(reg_x, reg5_y)
    print(f'Model 1: slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
    print(f'Model 2: slope: {slope2} intercept: {intercept2} p-value: {p_value2} R-squared: {r_value2 ** 2}')
    print(f'Model 3: slope: {slope3} intercept: {intercept3} p-value: {p_value3} R-squared: {r_value3 ** 2}')
    print(f'Model 4: slope: {slope4} intercept: {intercept4} p-value: {p_value4} R-squared: {r_value4 ** 2}')
    print(f'Model 5: slope: {slope5} intercept: {intercept5} p-value: {p_value5} R-squared: {r_value5 ** 2}')

    print(f'Done 400s plot!')
    plt.close()

    return fig


# Plots only xml files named ....x4....
def female_cross_norm_plot(input_df):
    print(f'Starting female plot...')

    sns.set_style("white")
    # Reading in txt file as dataframe
    data = pd.read_csv(input_df, sep="\t")
    # Sorting through file names
    data = data[data['File'].str.contains('x4')]
    # Plotting begins
    fig, ax = plt.subplots()
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission2", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission3", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission4", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})
    sns.regplot(x="Normalized_Window_Mean", y="Percent_Transmission5", ax=ax, data=data, fit_reg=True,
                scatter_kws={"color": "darkmagenta", "alpha": 0.006, "s": 20}, line_kws={"color": "black"})

    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    plt.ylim(0, 1)
    ax.yaxis.grid(True)
    ax.set(yticks=[0, 0.25, 0.5, 0.75, 1])

    plt.title('Female MODEL 400s Cross Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
    plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
    plt.ylabel('Percent Transmission', fontsize=18, weight='bold')

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # Saving Figure
    fig.get_figure()
    fig.savefig('Female MODEL norm 400s Cross Plot.png', bbox_inches="tight")
    # Calculating regression for each model
    reg_x = data['Normalized_Window_Mean'].values
    reg2_y = data['Percent_Transmission2'].values
    reg3_y = data['Percent_Transmission3'].values
    reg4_y = data['Percent_Transmission4'].values
    reg5_y = data['Percent_Transmission5'].values
    reg_y = data['Percent_Transmission'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(reg_x, reg2_y)
    slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(reg_x, reg3_y)
    slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(reg_x, reg4_y)
    slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(reg_x, reg5_y)
    print(f'Model 1: slope: {slope} intercept: {intercept} p-value: {p_value} R-squared: {r_value ** 2}')
    print(f'Model 2: slope: {slope2} intercept: {intercept2} p-value: {p_value2} R-squared: {r_value2 ** 2}')
    print(f'Model 3: slope: {slope3} intercept: {intercept3} p-value: {p_value3} R-squared: {r_value3 ** 2}')
    print(f'Model 4: slope: {slope4} intercept: {intercept4} p-value: {p_value4} R-squared: {r_value4 ** 2}')
    print(f'Model 5: slope: {slope5} intercept: {intercept5} p-value: {p_value5} R-squared: {r_value5 ** 2}')

    print(f'Done female plot!')
    plt.close()

    return fig

def main():
    if args.n:
        everything_norm_everything_graph(args.input_df)
        only400s_norm_plot(args.input_df)
        female_cross_norm_plot(args.input_df)
    else:
        everything_everything_graph(args.input_df)
        only400s_plot(args.input_df)
        female_cross_plot(args.input_df)

if __name__ == '__main__':
    main()