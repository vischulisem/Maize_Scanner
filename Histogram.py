#!/usr/local/bin/python3

# This script creates 3 histograms (rsquared, slope, p-value) for the models generated in Model.py
# Uses model_meta(normalized or not)_df.txt

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

def maybeionlylike400(input_whatever, otherthang):

    whatever_df = pd.read_csv(input_whatever, sep="\t")
    # Sorting through file names
    whatever_df = whatever_df[whatever_df['File_Name'].str.contains(r'x4..')]
    whatever_df = whatever_df[whatever_df['File_Name'].str.contains(r'X4..')]

    otherthang_df = pd.read_csv(otherthang, sep="\t")
    otherthang_df = otherthang_df[otherthang_df['File_Name'].str.contains(r'x4..')]
    otherthang_df = otherthang_df[otherthang_df['File_Name'].str.contains(r'X4..')]

    return whatever_df, otherthang_df

def this492familyiswack(model, normal):
    normal_df = pd.read_csv(normal, sep="\t")
    # Sorting through file names
    normal_df = normal_df[normal_df['File_Name'].str.contains(r'x492')]
    normal_df = normal_df[normal_df['File_Name'].str.contains(r'X4..')]

    model_df = pd.read_csv(model, sep="\t")
    model_df = model_df[model_df['File_Name'].str.contains(r'x492')]
    model_df = model_df[model_df['File_Name'].str.contains(r'X4..')]

    return model_df, normal_df


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
    results_dir = os.path.join(script_dir, 'Output_Histogram_Model/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_Slope_Hist.png', bbox_inches="tight")
    plt.close()
    return sub_data

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
    results_dir = os.path.join(script_dir, 'Output_Histogram_Model/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_Pvalue_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

def make_xml_r_hist(input_df, path):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")

    # Plotting hist without kde
    ax = sns.distplot(data['R-Squared'], kde=False, color='green', bins=50)
    plt.xlabel('R-Squared', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')

    plt.title('Normalized XML R-Squared Values', fontsize=25, fontweight='bold')

    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_allxml/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'XML_Rsq_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

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

    plt.title('Normalized XML Slope Values', fontsize=25, fontweight='bold')

    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_allxml/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'XML_Slope_Hist.png', bbox_inches="tight")
    plt.close()
    return data

def make_xml_pval_hist(input_df, path):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")

    ax = sns.distplot(data['P-Value'], kde=False, color='green', hist_kws={'log': True}, bins=50)
    plt.xlabel('P-Value', fontsize=18, fontweight='bold')
    plt.ylabel('Log Frequency', fontsize=18, fontweight='bold')

    plt.title('Normalized XML P-Values', fontsize=25, fontweight='bold')

    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_allxml/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'XML_Pvalue_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

# T test for model and all xml slopes because that's all anyone cares about tbh
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


def make4_r_hist(data, path):
    sns.set_style("white")

    sub_data = data[['R-Squared', 'r_value2', 'r_value3', 'r_value4', 'r_value5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    # Plotting hist without kde
    ax = sns.distplot(sub_data, kde=False, color='blue', bins=50)
    plt.xlabel('R-Squared', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')

    plt.title('Normalized 400s Model R-Squared Values', fontsize=25, fontweight='bold')

    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_Model/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_400sRsq_Hist.png', bbox_inches="tight")
    plt.close()
    return plt


def make4_slope_hist(data, path):
    sns.set_style("white")

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

    plt.title('Normalized 400s Model Slope Values', fontsize=25, fontweight='bold')

    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_Model/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_400sSlope_Hist.png', bbox_inches="tight")
    plt.close()
    return sub_data


def make4_pval_hist(data, path):
    sns.set_style("white")

    sub_data = data[['P-Value', 'p_value2', 'p_value3', 'p_value4', 'p_value5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    ax = sns.distplot(sub_data, kde=False, color='blue', hist_kws={'log': True}, bins=50)
    plt.xlabel('P-Value', fontsize=18, fontweight='bold')
    plt.ylabel('Log Frequency', fontsize=18, fontweight='bold')

    plt.title('Normalized 400s Model P-Values', fontsize=25, fontweight='bold')

    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_Model/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_400sPvalue_Hist.png', bbox_inches="tight")
    plt.close()
    return plt


def make4_xml_r_hist(data, path):
    sns.set_style("white")

    # Plotting hist without kde
    ax = sns.distplot(data['R-Squared'], kde=False, color='green', bins=50)
    plt.xlabel('R-Squared', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')

    plt.title('Normalized 400s XML R-Squared Values', fontsize=25, fontweight='bold')

    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_allxml/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'XML_400sRsq_Hist.png', bbox_inches="tight")
    plt.close()
    return plt


def make4_xml_slope_hist(data, path):
    sns.set_style("white")

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

    plt.title('Normalized 400s XML Slope Values', fontsize=25, fontweight='bold')

    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_allxml/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'XML_400sSlope_Hist.png', bbox_inches="tight")
    plt.close()
    return data


def make4_xml_pval_hist(data, path):
    sns.set_style("white")

    ax = sns.distplot(data['P-Value'], kde=False, color='green', hist_kws={'log': True}, bins=50)
    plt.xlabel('P-Value', fontsize=18, fontweight='bold')
    plt.ylabel('Log Frequency', fontsize=18, fontweight='bold')

    plt.title('Normalized 400s XML P-Values', fontsize=25, fontweight='bold')

    s_sav = ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_allxml/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'XML_400sPvalue_Hist.png', bbox_inches="tight")
    plt.close()
    return plt

def main():
    if args.f:
        model4_df, input4_allxml_df = maybeionlylike400(args.model_df, args.input_allxml_df)
        make4_r_hist(model4_df, args.path)
        model_df = make4_slope_hist(model4_df, args.path)
        make4_pval_hist(model4_df, args.path)
        make4_xml_r_hist(input4_allxml_df, args.path)
        all_slope_df = make4_xml_slope_hist(input4_allxml_df, args.path)
        make4_xml_pval_hist(input4_allxml_df, args.path)
        yay_ttest(all_slope_df, model_df)
    elif args.s:
        model4_df, input4_allxml_df = this492familyiswack(args.model_df, args.input_allxml_df)
        make4_r_hist(model4_df, args.path)
        model_df = make4_slope_hist(model4_df, args.path)
        make4_pval_hist(model4_df, args.path)
        make4_xml_r_hist(input4_allxml_df, args.path)
        all_slope_df = make4_xml_slope_hist(input4_allxml_df, args.path)
        make4_xml_pval_hist(input4_allxml_df, args.path)
        yay_ttest(all_slope_df, model_df)
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

