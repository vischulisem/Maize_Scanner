#!/usr/local/bin/python3

# This script creates histograms grouping families with known transmission defects or no defects
# Creates for both normal xmls and the model
# Outputs are graphs, changing path where plots are saved can be altered with -p

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
parser.add_argument('-t', '--table8', metavar='', help='Supplemental table 8 for transmission data.', type=str)
parser.add_argument('-p', '--path', metavar='', help='List path where you want files saved to.', default=os.getcwd(), type=str)
args = parser.parse_args()

# Makes histogram of slopes of all xml files with no transmission defect
def nodefect_values(table8, meta_df, path):
    # Reading table 8 into dataframe
    eight_df = pd.read_csv(table8, sep="\t")
    # Removing female crosses
    eight_df = eight_df[eight_df['Mutant allele parent'] != 'female']
    eight_df = eight_df.reset_index(drop=True)
    # Selecting only columns we're interested in
    eight_df = eight_df.loc[:, ['Tracking number', 'Expression class', 'Adjusted p-value']]
    eight_df = eight_df.reset_index(drop=True)

    # Choose when pvalue greater than 0.05
    eight_df = eight_df[eight_df['Adjusted p-value'] > 0.05]
    eight_df = eight_df.reset_index(drop=True)

    # Reading meta df in as dataframe
    xml_df = pd.read_csv(meta_df, sep="\t")
    # Converting file names to strings for reg ex
    xml_df['File_Name'] = xml_df['File_Name'].astype(str)

    # In table 8, put all file numbers to a list and use them to search
    # the meta df for file names with the same numbers
    search_values = eight_df['Tracking number'].tolist()
    search_values = ['x' + str(i) for i in search_values]
    xml_df = xml_df[xml_df.File_Name.str.contains('|'.join(search_values))]

    # Plot these families in xml_df
    sns.set_style("white")

    # Plotting hist without kde
    ax = sns.distplot(xml_df['Slope'], kde=False, color='green', bins=50)
    plt.xlabel('Slope', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    # Creating another Y axis
    second_ax = ax.twinx()

    # Plotting kde without hist on the second Y axis
    sns.distplot(xml_df['Slope'], ax=second_ax, kde=True, hist=False, color='black')

    # Removing Y ticks from the second axis
    second_ax.set_yticks([])

    plt.title('No Defect Slope Values', fontsize=25, fontweight='bold')
    # Saving figure
    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_defect/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'No_defect_Slope_Hist.png', bbox_inches="tight")
    plt.close()
    return xml_df

# Making histogram of slopes of all xml files with known transmission defect
def defect_values(table8, meta_df, path):
    # Read in table 8 as df
    deight_df = pd.read_csv(table8, sep="\t")
    # Remove columns with female crosses
    deight_df = deight_df[deight_df['Mutant allele parent'] != 'female']
    deight_df = deight_df.reset_index(drop=True)
    # Only look at these columns
    deight_df = deight_df.loc[:, ['Tracking number', 'Expression class', 'Adjusted p-value']]
    deight_df = deight_df.reset_index(drop=True)

    # Choose when pvalue greater than 0.05
    deight_df = deight_df[deight_df['Adjusted p-value'] <= 0.05]
    deight_df = deight_df.reset_index(drop=True)
    # Read meta df into dataframe
    dxml_df = pd.read_csv(meta_df, sep="\t")
    # Convert file names to strings for reg ex
    dxml_df['File_Name'] = dxml_df['File_Name'].astype(str)
    # Put tracking numbers in a list from table 8 and put them together with 'x'
    search_values = deight_df['Tracking number'].tolist()
    search_values = ['x' + str(i) for i in search_values]
    # Search meta dataframe for these families
    dxml_df = dxml_df[dxml_df.File_Name.str.contains('|'.join(search_values))]

    # Start plotting
    sns.set_style("white")

    # Plotting hist without kde
    ax = sns.distplot(dxml_df['Slope'], kde=False, color='green', bins=50)
    plt.xlabel('Slope', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    # Creating another Y axis
    second_ax = ax.twinx()

    # Plotting kde without hist on the second Y axis
    sns.distplot(dxml_df['Slope'], ax=second_ax, kde=True, hist=False, color='black')

    # Removing Y ticks from the second axis
    second_ax.set_yticks([])

    plt.title('Defect Slope Values', fontsize=25, fontweight='bold')
    # Saving figure
    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_defect/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Defect_Slope_Hist.png', bbox_inches="tight")
    plt.close()
    return dxml_df

# Make histogram of all slopes from model df with no defect
def model_nodefect_values(table8, model_df, path):
    # Reading in table 8 as dataframe
    meight_df = pd.read_csv(table8, sep="\t")
    # Removing rows with female crosses
    meight_df = meight_df[meight_df['Mutant allele parent'] != 'female']
    meight_df = meight_df.reset_index(drop=True)
    # Only looking at columns we're interested in
    meight_df = meight_df.loc[:, ['Tracking number', 'Expression class', 'Adjusted p-value']]
    meight_df = meight_df.reset_index(drop=True)

    # Choose when pvalue greater than 0.05
    meight_df = meight_df[meight_df['Adjusted p-value'] > 0.05]
    meight_df = meight_df.reset_index(drop=True)

    # Reading in model meta txt to dataframe
    model_meta_df = pd.read_csv(model_df, sep="\t")
    # Converting file names to strings for reg ex
    model_meta_df['File_Name'] = model_meta_df['File_Name'].astype(str)
    # Searching table 8 df for tracking numbers and putting into list
    search_values = meight_df['Tracking number'].tolist()
    # Piece each number with 'x'
    search_values = ['x' + str(i) for i in search_values]
    # Search model meta df for these families
    model_meta_df = model_meta_df[model_meta_df.File_Name.str.contains('|'.join(search_values))]

    # Stack all 5 model slopes on top of each other for plotting
    sub_data = model_meta_df[['Slope', 'slope2', 'slope3', 'slope4', 'slope5']]
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

    plt.title('No Defect Model Slope Values', fontsize=25, fontweight='bold')
    # Saving figure
    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_defect/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_nodefect_Slope_Hist.png', bbox_inches="tight")
    plt.close()
    return sub_data

def model_defect_values(table8, model_df, path):
    # Read table 8 in as dataframe
    mdeight_df = pd.read_csv(table8, sep="\t")
    # Remove rows with female crosses
    mdeight_df = mdeight_df[mdeight_df['Mutant allele parent'] != 'female']
    mdeight_df = mdeight_df.reset_index(drop=True)
    # Only looking at these rows we're interested in
    mdeight_df = mdeight_df.loc[:, ['Tracking number', 'Expression class', 'Adjusted p-value']]
    mdeight_df = mdeight_df.reset_index(drop=True)

    # Choose when pvalue greater than 0.05
    mdeight_df = mdeight_df[mdeight_df['Adjusted p-value'] <= 0.05]
    mdeight_df = mdeight_df.reset_index(drop=True)

    # Reading in model meta txt as df
    model_meta_df = pd.read_csv(model_df, sep="\t")
    model_meta_df['File_Name'] = model_meta_df['File_Name'].astype(str)
    # Putting tracking numbers in a list
    search_values = mdeight_df['Tracking number'].tolist()
    # For each value put x with value
    search_values = ['x' + str(i) for i in search_values]
    # Search model meta df for these families
    model_meta_df = model_meta_df[model_meta_df.File_Name.str.contains('|'.join(search_values))]
    # Stack 5 slopes on top of each other for plotting
    d_sub_data = model_meta_df[['Slope', 'slope2', 'slope3', 'slope4', 'slope5']]
    d_sub_data = d_sub_data.unstack().reset_index(drop=True)
    # Plotting hist without kde
    ax = sns.distplot(d_sub_data, kde=False, color='blue', bins=50)
    plt.xlabel('Slope', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    # Creating another Y axis
    second_ax = ax.twinx()

    # Plotting kde without hist on the second Y axis
    sns.distplot(d_sub_data, ax=second_ax, kde=True, hist=False, color='red')

    # Removing Y ticks from the second axis
    second_ax.set_yticks([])

    plt.title('Defect Model Slope Values', fontsize=25, fontweight='bold')
    # Save figure
    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_defect/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_defect_Slope_Hist.png', bbox_inches="tight")
    plt.close()
    return d_sub_data

# T test for model and all xml slopes because that's all anyone cares about tbh
# Results printed in terminal
def yay_ttest(d_sub_data, sub_data, dxml_df, xml_df):
    xmlslope = xml_df['Slope'].values
    modelslope = sub_data.values
    teeheetest = stats.ttest_ind(xmlslope, modelslope)
    print(f'No defect result: {teeheetest}')
    xmlmean = xml_df['Slope'].mean()
    print(f'XML no defect slope mean: {xmlmean}')
    modelmean = sub_data.mean()
    print(f'Model no defect slope mean: {modelmean}')

    dxmlslope = dxml_df['Slope'].values
    dmodelslope = d_sub_data.values
    dteeheetest = stats.ttest_ind(dxmlslope, dmodelslope)
    print(f'Defect result: {dteeheetest}')
    dxmlmean = dxml_df['Slope'].mean()
    print(f'XML defect slope mean: {dxmlmean}')
    dmodelmean = d_sub_data.mean()
    print(f'Model defect slope mean: {dmodelmean}')
    return d_sub_data

# Main function to run everything
def main():
    xml_df = nodefect_values(args.table8, args.input_df, args.path)
    dxml_df = defect_values(args.table8, args.input_df, args.path)
    sub_data = model_nodefect_values(args.table8, args.model_df, args.path)
    dsub_data = model_defect_values(args.table8, args.model_df, args.path)
    yay_ttest(xml_df, dxml_df, sub_data, dsub_data)

if __name__ == '__main__':
    main()

