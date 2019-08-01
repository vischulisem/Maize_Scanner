#!/usr/local/bin/python3

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def make_r_hist(input_df):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")
    sub_data = data[['R-Squared', 'r_value2', 'r_value3', 'r_value4', 'r_value5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    fig, ax = plt.subplots()
    r_plot = sns.distplot(sub_data, color='red', kde=True, bins=100)
    ax.set(xlim=(0, 1))
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    plt.title('Model R-Squared Values', fontsize=18)
    plt.xlabel('R-Squared', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)

    r_sav = r_plot.get_figure()
    r_sav.savefig('Model_R_Hist.png')
    plt.close()
    return fig

def make_slope_hist(input_df):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")
    sub_data = data[['Slope', 'slope2', 'slope3', 'slope4', 'slope5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    fig, ax = plt.subplots()
    s_plot = sns.distplot(sub_data, color='blue', kde=True, bins=100)

    plt.title('Model Slope Values', fontsize=18)
    plt.xlabel('Slope', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    s_sav = s_plot.get_figure()
    s_sav.savefig('Model_Slope_Hist.png')
    plt.close()
    return fig

def make_pval_hist(input_df):
    sns.set_style("white")

    data = pd.read_csv(input_df, sep="\t")
    sub_data = data[['P-Value', 'p_value2', 'p_value3', 'p_value4', 'p_value5']]
    sub_data = sub_data.unstack().reset_index(drop=True)

    fig, ax = plt.subplots()
    s_plot = sns.distplot(sub_data, color='green', kde=True, bins=50)

    plt.title('Model P-Values', fontsize=18)
    plt.xlabel('P-Values', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.show()
    # s_sav = s_plot.get_figure()
    # s_sav.savefig('Model_Slope_Hist.png')
    # plt.close()
    return fig

plot = make_r_hist("/Users/elysevischulis/PycharmProjects/Maize_Scanner/meta_model_df.txt")
plot2 = make_slope_hist("/Users/elysevischulis/PycharmProjects/Maize_Scanner/meta_model_df.txt")
plot3 = make_pval_hist("/Users/elysevischulis/PycharmProjects/Maize_Scanner/meta_model_df.txt")