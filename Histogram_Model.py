#!/usr/local/bin/python3

# This script creates 3 histograms (rsquared, slope, p-value) for the models generated in Model.py
# Uses model_meta(normalized or not)_df.txt

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Setting up argparse arguments
parser = argparse.ArgumentParser(description='Given model meta df, returns histograms for regression stats.')
parser.add_argument('-i', '--input_df', metavar='', help='Input meta dataframe filename.', type=str)
parser.add_argument('-p', '--path', metavar='', help='List path where you want files saved to.', default=os.getcwd(), type=str)
args = parser.parse_args()


def make_r_hist(input_df, path):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")
    sub_data = data[['R-Squared', 'r_value2', 'r_value3', 'r_value4', 'r_value5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    # Plotting hist without kde
    ax = sns.distplot(sub_data, kde=False, color='blue')
    plt.xlabel('R-Squared', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')

    plt.title('Normalized Model R-Squared Values', fontsize=25, fontweight='bold')

    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_Model/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_Rsq_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

def make_slope_hist(input_df, path):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")
    sub_data = data[['Slope', 'slope2', 'slope3', 'slope4', 'slope5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    # Plotting hist without kde
    ax = sns.distplot(sub_data, kde=False, color='blue')
    plt.xlabel('Slope', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    # Creating another Y axis
    second_ax = ax.twinx()

    # Plotting kde without hist on the second Y axis
    sns.distplot(sub_data, ax=second_ax, kde=True, hist=False, color='red')

    # Removing Y ticks from the second axis
    second_ax.set_yticks([])

    plt.title('Normalized Model Slope Values', fontsize=25, fontweight='bold')

    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_Model/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_Slope_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

def make_pval_hist(input_df, path):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")
    sub_data = data[['P-Value', 'p_value2', 'p_value3', 'p_value4', 'p_value5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    ax = sns.distplot(sub_data, kde=False, color='blue', hist_kws={'log': True})
    plt.xlabel('P-Value', fontsize=18, fontweight='bold')
    plt.ylabel('Log Frequency', fontsize=18, fontweight='bold')

    plt.title('Normalized Model P-Values', fontsize=25, fontweight='bold')

    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_Model/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_Pvalue_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

def main():
    make_r_hist(args.input_df, args.path)
    make_slope_hist(args.input_df, args.path)
    make_pval_hist(args.input_df, args.path)

if __name__ == '__main__':
    main()