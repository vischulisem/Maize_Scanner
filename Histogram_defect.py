#!/usr/local/bin/python3

import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy import stats

def iwannaseeeverythingsortedok(meta_df):
    big_df = pd.read_csv(meta_df, sep="\t")
    big_df = big_df.sort_values(by=['Slope'])
    big_df = big_df.reset_index(drop=True)
    print(f'lowest slopes: {big_df.File_Name.head(15)}')
    print(f'highest slopes: {big_df.File_Name.tail(15)}')
    return big_df

def nodefect_values(table8, meta_df, path):
    eight_df = pd.read_csv(table8, sep="\t")

    eight_df = eight_df[eight_df['Mutant allele parent'] != 'female']
    eight_df = eight_df.reset_index(drop=True)
    eight_df = eight_df.loc[:, ['Tracking number', 'Expression class', 'Adjusted p-value']]
    eight_df = eight_df.reset_index(drop=True)

    # Choose when pvalue greater than 0.05
    eight_df = eight_df[eight_df['Adjusted p-value'] > 0.05]
    eight_df = eight_df.reset_index(drop=True)

    xml_df = pd.read_csv(meta_df, sep="\t")
    xml_df['File_Name'] = xml_df['File_Name'].astype(str)

    search_values = eight_df['Tracking number'].tolist()
    search_values = ['x' + str(i) for i in search_values]
    xml_df = xml_df[xml_df.File_Name.str.contains('|'.join(search_values))]

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

    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_defect/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'No_defect_Slope_Hist.png', bbox_inches="tight")
    plt.close()

    xml_df = xml_df.sort_values(by=['Slope'])
    xml_df = xml_df.reset_index(drop=True)
    print(f'No defect: lowest slopes: {xml_df.File_Name.head(11)}')
    print(f'No defect: highest slopes: {xml_df.File_Name.tail(11)}')
    return eight_df, xml_df

def defect_values(table8, meta_df, path):
    deight_df = pd.read_csv(table8, sep="\t")

    deight_df = deight_df[deight_df['Mutant allele parent'] != 'female']
    deight_df = deight_df.reset_index(drop=True)
    deight_df = deight_df.loc[:, ['Tracking number', 'Expression class', 'Adjusted p-value']]
    deight_df = deight_df.reset_index(drop=True)

    # Choose when pvalue greater than 0.05
    deight_df = deight_df[deight_df['Adjusted p-value'] <= 0.05]
    deight_df = deight_df.reset_index(drop=True)

    dxml_df = pd.read_csv(meta_df, sep="\t")
    dxml_df['File_Name'] = dxml_df['File_Name'].astype(str)

    search_values = deight_df['Tracking number'].tolist()
    search_values = ['x' + str(i) for i in search_values]
    dxml_df = dxml_df[dxml_df.File_Name.str.contains('|'.join(search_values))]

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

    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_defect/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Defect_Slope_Hist.png', bbox_inches="tight")
    plt.close()
    dxml_df = dxml_df.sort_values(by=['Slope'])
    dxml_df = dxml_df.reset_index(drop=True)

    print(f'Defect: lowest slopes: {dxml_df.File_Name.head(11)}')
    print(f'Defect: highest slopes: {dxml_df.File_Name.tail(11)}')
    return dxml_df, deight_df

def model_nodefect_values(table8, model_df, path):
    meight_df = pd.read_csv(table8, sep="\t")

    meight_df = meight_df[meight_df['Mutant allele parent'] != 'female']
    meight_df = meight_df.reset_index(drop=True)
    meight_df = meight_df.loc[:, ['Tracking number', 'Expression class', 'Adjusted p-value']]
    meight_df = meight_df.reset_index(drop=True)

    # Choose when pvalue greater than 0.05
    meight_df = meight_df[meight_df['Adjusted p-value'] > 0.05]
    meight_df = meight_df.reset_index(drop=True)

    model_meta_df = pd.read_csv(model_df, sep="\t")
    model_meta_df['File_Name'] = model_meta_df['File_Name'].astype(str)

    search_values = meight_df['Tracking number'].tolist()
    search_values = ['x' + str(i) for i in search_values]
    model_meta_df = model_meta_df[model_meta_df.File_Name.str.contains('|'.join(search_values))]

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

    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_defect/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_nodefect_Slope_Hist.png', bbox_inches="tight")
    plt.close()
    return sub_data

def model_defect_values(table8, model_df, path):
    mdeight_df = pd.read_csv(table8, sep="\t")

    mdeight_df = mdeight_df[mdeight_df['Mutant allele parent'] != 'female']
    mdeight_df = mdeight_df.reset_index(drop=True)
    mdeight_df = mdeight_df.loc[:, ['Tracking number', 'Expression class', 'Adjusted p-value']]
    mdeight_df = mdeight_df.reset_index(drop=True)

    # Choose when pvalue greater than 0.05
    mdeight_df = mdeight_df[mdeight_df['Adjusted p-value'] <= 0.05]
    mdeight_df = mdeight_df.reset_index(drop=True)

    model_meta_df = pd.read_csv(model_df, sep="\t")
    model_meta_df['File_Name'] = model_meta_df['File_Name'].astype(str)

    search_values = mdeight_df['Tracking number'].tolist()
    search_values = ['x' + str(i) for i in search_values]
    model_meta_df = model_meta_df[model_meta_df.File_Name.str.contains('|'.join(search_values))]

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

    s_sav = second_ax.get_figure()
    script_dir = path
    results_dir = os.path.join(script_dir, 'Output_Histogram_defect/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    s_sav.savefig(results_dir + 'Model_defect_Slope_Hist.png', bbox_inches="tight")
    plt.close()
    return d_sub_data

# T test for model and all xml slopes because that's all anyone cares about tbh
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

big_df = iwannaseeeverythingsortedok("/Users/elysevischulis/PycharmProjects/Maize_Scanner/meta_normalized_df.txt")
eight_df, xml_df = nodefect_values("/Users/elysevischulis/PycharmProjects/Maize_Scanner/Data_for_Analysis/Table8.tsv", "/Users/elysevischulis/PycharmProjects/Maize_Scanner/meta_normalized_df.txt", "/Users/elysevischulis/PycharmProjects/Maize_Scanner")
dxml_df, deight_df = defect_values("/Users/elysevischulis/PycharmProjects/Maize_Scanner/Data_for_Analysis/Table8.tsv", "/Users/elysevischulis/PycharmProjects/Maize_Scanner/meta_normalized_df.txt", "/Users/elysevischulis/PycharmProjects/Maize_Scanner")
sub_data = model_nodefect_values("/Users/elysevischulis/PycharmProjects/Maize_Scanner/Data_for_Analysis/Table8.tsv", "/Users/elysevischulis/PycharmProjects/Maize_Scanner/meta_normalized_model_df.txt", "/Users/elysevischulis/PycharmProjects/Maize_Scanner")
d_sub_data = model_defect_values("/Users/elysevischulis/PycharmProjects/Maize_Scanner/Data_for_Analysis/Table8.tsv", "/Users/elysevischulis/PycharmProjects/Maize_Scanner/meta_normalized_model_df.txt", "/Users/elysevischulis/PycharmProjects/Maize_Scanner")
yay_ttest(d_sub_data, sub_data, dxml_df, xml_df)