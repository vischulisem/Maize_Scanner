#!/usr/local/bin/python3

# This script creates 3 histograms (r-squared, slope, p-value) for the models generated in Model.py
# Uses model_meta(normalized or not)_df.txt
# Also creates 3 histograms for normal xml files from meta_normalized_df.txt
# Can determine path where plots are saved with -p
# If use -f will only make with cross in 400s (X4..x4..)
# If -s then will only make histogram for 492 family

import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy import stats

# Setting up argparse arguments
parser = argparse.ArgumentParser(description='Given model meta df and normal meta df returns histograms for regression stats.')
parser.add_argument('-m', '--model_df', metavar='', help='Input Model meta dataframe filename.', type=str)
parser.add_argument('-i', '--input_allxml_df', metavar='', help='Input all xml meta dataframe filename.', type=str)
parser.add_argument('-p', '--path', metavar='', help='List path where you want files saved to.', default=os.getcwd(), type=str)
parser.add_argument('-f', action='store_true', help='If present, will only make histogram of X4..x4.. families.')
parser.add_argument('-s', action='store_true', help='If present, will only make histogram of x492 family.')
args = parser.parse_args()

# Function for sorting through both txt files for only 400s files
# Returns sorted dataframes
def maybeionlylike400(input_whatever, otherthang):

    whatever_df = pd.read_csv(input_whatever, sep="\t")
    # Sorting through file names
    whatever_df = whatever_df[whatever_df['File_Name'].str.contains(r'x4..')]
    whatever_df = whatever_df[whatever_df['File_Name'].str.contains(r'X4..')]

    otherthang_df = pd.read_csv(otherthang, sep="\t")
    otherthang_df = otherthang_df[otherthang_df['File_Name'].str.contains(r'x4..')]
    otherthang_df = otherthang_df[otherthang_df['File_Name'].str.contains(r'X4..')]

    return whatever_df, otherthang_df

# Function for sorting through both txt files for only 492s files
# Returns sorted dataframes
def this492familyiswack(model, normal):
    normal_df = pd.read_csv(normal, sep="\t")
    # Sorting through file names
    normal_df = normal_df[normal_df['File_Name'].str.contains(r'x492')]
    normal_df = normal_df[normal_df['File_Name'].str.contains(r'X4..')]

    model_df = pd.read_csv(model, sep="\t")
    model_df = model_df[model_df['File_Name'].str.contains(r'x492')]
    model_df = model_df[model_df['File_Name'].str.contains(r'X4..')]

    return model_df, normal_df

# Model histogram R squared
def make_r_hist(input_df, path):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")
    sub_data = data[['R-Squared', 'r_value2', 'r_value3', 'r_value4', 'r_value5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    # Plotting hist without kde
    ax = sns.distplot(sub_data, kde=False, color='blue', bins=50)
    plt.xlabel('R-Squared', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')

    plt.title('Normalized Model R-Squared Values', fontsize=25, fontweight='bold')

    # Saving figure
    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_Rsq_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

# Model histogram of slope
def make_slope_hist(input_df, path):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")
    sub_data = data[['Slope', 'slope2', 'slope3', 'slope4', 'slope5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    # Plotting hist without kde
    ax = sns.distplot(sub_data, kde=False, color='blue', bins=50)
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
    results_dir = os.path.join(script_dir, 'Output_Histogram/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_Slope_Hist.png', bbox_inches="tight")
    plt.close()
    return sub_data

# Model histogram for p-value
def make_pval_hist(input_df, path):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")
    sub_data = data[['P-Value', 'p_value2', 'p_value3', 'p_value4', 'p_value5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    ax = sns.distplot(sub_data, kde=False, color='blue', hist_kws={'log': True}, bins=50)
    plt.xlabel('P-Value', fontsize=18, fontweight='bold')
    plt.ylabel('Log Frequency', fontsize=18, fontweight='bold')

    plt.title('Normalized Model P-Values', fontsize=25, fontweight='bold')

    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_Pvalue_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

# Normal xml histogram of R-squared
def make_xml_r_hist(input_df, path):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")

    # Plotting hist without kde
    ax = sns.distplot(data['R-Squared'], kde=False, color='green', bins=50)
    plt.xlabel('R-Squared', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')

    plt.title('Normalized Actual R-Squared Values', fontsize=25, fontweight='bold')

    # Saving figure
    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'XML_Rsq_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

# Normal xml histogram of slope
def make_xml_slope_hist(input_df, path):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")

    # Plotting hist without kde
    ax = sns.distplot(data['Slope'], kde=False, color='green', bins=50)
    plt.xlabel('Slope', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    # Creating another Y axis
    second_ax = ax.twinx()

    # Plotting kde without hist on the second Y axis
    sns.distplot(data['Slope'], ax=second_ax, kde=True, hist=False, color='black')

    # Removing Y ticks from the second axis
    second_ax.set_yticks([])

    plt.title('Normalized Actual Slope Values', fontsize=25, fontweight='bold')

    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'XML_Slope_Hist.png', bbox_inches="tight")
    plt.close()
    return data

# Normal xml histogram of p-value
def make_xml_pval_hist(input_df, path):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")

    ax = sns.distplot(data['P-Value'], kde=False, color='green', hist_kws={'log': True}, bins=50)
    plt.xlabel('P-Value', fontsize=18, fontweight='bold')
    plt.ylabel('Log Frequency', fontsize=18, fontweight='bold')

    plt.title('Normalized Actual P-Values', fontsize=25, fontweight='bold')

    # Saving figure
    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'XML_Pvalue_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

# T test for model and all xml slopes because that's all anyone cares about tbh
# Will print results in terminal
def yay_ttest(xml_df, model_df):
    xmlslope = xml_df['Slope'].values
    modelslope = model_df.values
    teeheetest = stats.ttest_ind(xmlslope, modelslope)
    print(teeheetest)
    xmlmean = xml_df['Slope'].mean()
    print(f'XML slope mean: {xmlmean}')
    modelmean = model_df.mean()
    print(f'Model slope mean: {modelmean}')
    return model_df

# Make histogram if sorted by 400s or 492
# Model r squared
def make4_r_hist(data, path, argss):
    sns.set_style("white")

    sub_data = data[['R-Squared', 'r_value2', 'r_value3', 'r_value4', 'r_value5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    # Plotting hist without kde
    ax = sns.distplot(sub_data, kde=False, color='blue', bins=50)
    plt.xlabel('R-Squared', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')

    # IF sorted by 492, the title will be
    if argss:
        plt.title('Normalized 492s Model R-Squared Values', fontsize=25, fontweight='bold')
    # Otherwise this is the title
    else:
        plt.title('Normalized 400s Model R-Squared Values', fontsize=25, fontweight='bold')

    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # Saving figure if 492
    if argss:
        s_sav.savefig(results_dir + 'Model_492sRsq_Hist.png', bbox_inches="tight")
    # Otherwise this is the name
    else:
        s_sav.savefig(results_dir + 'Model_400sRsq_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

# Make histogram if sorted by 400s or 492
# Model slope
def make4_slope_hist(data, path, argss):
    sns.set_style("white")

    sub_data = data[['Slope', 'slope2', 'slope3', 'slope4', 'slope5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    # Plotting hist without kde
    ax = sns.distplot(sub_data, kde=False, color='blue', bins=50)
    plt.xlabel('Slope', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    # Setting axis lims if 492 for comparing
    if argss:
        plt.ylim(0, 2)
        plt.xlim(-0.4, 0.4)
    # Creating another Y axis
    second_ax = ax.twinx()

    # Plotting kde without hist on the second Y axis
    sns.distplot(sub_data, ax=second_ax, kde=True, hist=False, color='red')

    # Removing Y ticks from the second axis
    second_ax.set_yticks([])

    # If sorted for 492, title
    if argss:
        plt.title('Normalized 492s Model Slope Values', fontsize=25, fontweight='bold')
    # Otherwise sorted by 400s
    else:
        plt.title('Normalized 400s Model Slope Values', fontsize=25, fontweight='bold')

    # Saving figure
    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # If 492 sorted save as this
    if argss:
        s_sav.savefig(results_dir + 'Model_492sSlope_Hist.png', bbox_inches="tight")
    # If 400 sorted save as this
    else:
        s_sav.savefig(results_dir + 'Model_400sSlope_Hist.png', bbox_inches="tight")
    plt.close()
    return sub_data

# Make histogram if sorted by 400s or 492
# Model p-value
def make4_pval_hist(data, path, argss):
    sns.set_style("white")

    sub_data = data[['P-Value', 'p_value2', 'p_value3', 'p_value4', 'p_value5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    ax = sns.distplot(sub_data, kde=False, color='blue', hist_kws={'log': True}, bins=50)
    plt.xlabel('P-Value', fontsize=18, fontweight='bold')
    plt.ylabel('Log Frequency', fontsize=18, fontweight='bold')

    if argss:
        plt.title('Normalized 492s Model P-Values', fontsize=25, fontweight='bold')
    else:
        plt.title('Normalized 400s Model P-Values', fontsize=25, fontweight='bold')

    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if argss:
        s_sav.savefig(results_dir + 'Model_492sPvalue_Hist.png', bbox_inches="tight")
    else:
        s_sav.savefig(results_dir + 'Model_400sPvalue_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

# Make histogram if sorted by 400s or 492
# normal xml R-squared
def make4_xml_r_hist(data, path, argss):
    sns.set_style("white")

    # Plotting hist without kde
    ax = sns.distplot(data['R-Squared'], kde=False, color='green', bins=50)
    plt.xlabel('R-Squared', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')

    if argss:
        plt.title('Normalized 492s Actual R-Squared Values', fontsize=25, fontweight='bold')
    else:
        plt.title('Normalized 400s Actual R-Squared Values', fontsize=25, fontweight='bold')

    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if argss:
        s_sav.savefig(results_dir + 'XML_492sRsq_Hist.png', bbox_inches="tight")
    else:
        s_sav.savefig(results_dir + 'XML_400sRsq_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

# Make histogram if sorted by 400s or 492
# Normal xml slope
def make4_xml_slope_hist(data, path, argss):
    sns.set_style("white")

    # Plotting hist without kde
    ax = sns.distplot(data['Slope'], kde=False, color='green', bins=50)
    plt.xlabel('Slope', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    if argss:
        plt.ylim(0, 2)
        plt.xlim(-0.4, 0.4)
    # Creating another Y axis
    second_ax = ax.twinx()

    # Plotting kde without hist on the second Y axis
    sns.distplot(data['Slope'], ax=second_ax, kde=True, hist=False, color='black')

    # Removing Y ticks from the second axis
    second_ax.set_yticks([])

    if argss:
        plt.title('Normalized 492s Actual Slope Values', fontsize=25, fontweight='bold')
    else:
        plt.title('Normalized 400s Actual Slope Values', fontsize=25, fontweight='bold')

    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    if argss:
        s_sav.savefig(results_dir + 'XML_492sSlope_Hist.png', bbox_inches="tight")
    else:
        s_sav.savefig(results_dir + 'XML_400sSlope_Hist.png', bbox_inches="tight")
    plt.close()
    return data

# Make histogram if sorted by 400s or 492
# Normal xml p-value
def make4_xml_pval_hist(data, path, argss):
    sns.set_style("white")

    ax = sns.distplot(data['P-Value'], kde=False, color='green', hist_kws={'log': True}, bins=50)
    plt.xlabel('P-Value', fontsize=18, fontweight='bold')
    plt.ylabel('Log Frequency', fontsize=18, fontweight='bold')

    if argss:
        plt.title('Normalized 492s Actual P-Values', fontsize=25, fontweight='bold')
    else:
        plt.title('Normalized 400s Actual P-Values', fontsize=25, fontweight='bold')
    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if argss:
        s_sav.savefig(results_dir + 'XML_492sPvalue_Hist.png', bbox_inches="tight")
    else:
        s_sav.savefig(results_dir + 'XML_400sPvalue_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

# Main function to run everything
def main():
    # IF sort for everything in 400s
    if args.f:
        model4_df, input4_allxml_df = maybeionlylike400(args.model_df, args.input_allxml_df)
        make4_r_hist(model4_df, args.path, args.s)
        model_df = make4_slope_hist(model4_df, args.path, args.s)
        make4_pval_hist(model4_df, args.path, args.s)
        make4_xml_r_hist(input4_allxml_df, args.path, args.s)
        all_slope_df = make4_xml_slope_hist(input4_allxml_df, args.path, args.s)
        make4_xml_pval_hist(input4_allxml_df, args.path, args.s)
        yay_ttest(all_slope_df, model_df)
    # IF you only want sorted for 492s family
    elif args.s:
        model4_df, input4_allxml_df = this492familyiswack(args.model_df, args.input_allxml_df)
        make4_r_hist(model4_df, args.path, args.s)
        model_df = make4_slope_hist(model4_df, args.path, args.s)
        make4_pval_hist(model4_df, args.path, args.s)
        make4_xml_r_hist(input4_allxml_df, args.path, args.s)
        all_slope_df = make4_xml_slope_hist(input4_allxml_df, args.path, args.s)
        make4_xml_pval_hist(input4_allxml_df, args.path, args.s)
        yay_ttest(all_slope_df, model_df)
    # Otherwise just make histogram of everything in file
    else:
        make_r_hist(args.model_df, args.path)
        model_df = make_slope_hist(args.model_df, args.path)
        make_pval_hist(args.model_df, args.path)
        make_xml_r_hist(args.input_allxml_df, args.path)
        all_slope_df = make_xml_slope_hist(args.input_allxml_df, args.path)
        make_xml_pval_hist(args.input_allxml_df, args.path)
        yay_ttest(all_slope_df, model_df)

if __name__ == '__main__':
    main()

