# Overview of Maize Scanner
This repository is a group of scripts and data to analyze an ear of maize. Each script can be cloned and run in the terminal and uses argparse as argument parser.</br>
Scripts and the order they should be run are outlined below.

## Prerequisits
numpy https://docs.scipy.org/doc/numpy/user/install.html </br>
argparse https://pypi.org/project/argparse/</br>
xml.etree.ElementTree https://docs.python.org/3/library/xml.etree.elementtree.html</br>
pandas https://pandas.pydata.org/pandas-docs/stable/install.html</br>
matplotlib https://matplotlib.org/3.1.1/users/installing.html</br>
seaborn https://seaborn.pydata.org/installing.html</br>
pylab https://www.techwalla.com/articles/how-to-install-pylab-on-python</br>
scipy https://www.scipy.org/install.html</br>
pysal https://pysal.org/install</br>

## XML Analysis

### 1. XML_to_ChiSquareTrasmPlot.py 

  What it does:</br>
     Takes xml file, gets X,Y coordinate points for each kernel, and labels as fluorescent or nonfluorescent. 
    Creates sliding window parameter to scan ear counting the number of kernels, fluor, and nonfluor in each window.
    Calculates chi square statistic for entire ear and each individual window.
    Plots positional percent transmission line colored based on whether point is above or below p = 0.05 for chi squared test.
    Also plots and calculates regression.</br>
      This script accepts an xml file or directory of xml files, window width, and step size in pixels as arguments. Defaults are set
    for width and step size. Optional arguments listed:</br>
    -tk --total_kernels to adjust the minimum number of kernels per entire ear. Default set at 50. Therefore any xml files with 
    less than 50 kernels per ear are skipped.</br>
    -n will normalize the X coordinates "window_mean" for all graphs</br>
    -p --path designates the path to save files. Default is current path.</br>
    Outputs: New directory 'Transmission_plots' containing graphs corresponding to each xml file with positional
    percent transmission for each window, regression, and ear statistics. Also outputs 'meta_df.txt' which contains overall 
    ear stats and p-values. Also outputs 'everything_df.txt' which contains all window calculations and p values for each window for
    each xml file. .txt files are tab delimited use to reconvert to new pandas dataframe in later scripts. </br>
    
   Example of positional transmission plot:
   ![Positional Transmission Plot](Users/elysevischulis/PycharmProjects/MaizeScanner/Output_XML_to_ChiSquareTransmPlot/Transmission_Norm_plots/X4-6x402.png)
    
### 2. Family_Graphs.py

  What it does:</br>
      Takes 'everything_df.txt' and creates a variety of plots based on male/female crosses. </br>
    Plots all xml files on single scatterplot with regression line. </br>
    Plots all files with male and female in 400s on scatterplot with regression line.</br>
    Plots all females in 400s on scatterplot with regression line.</br>
    Plots each male family in 400s with one or more lines on each graph corresponding to each xml file in family. Also plots
    regression lines.</br>
      This script accepts everything_df.txt, starting value for male family plots (must be in 400s), and ending value for male
    family plots (must be in 400s). Example) 410 499. Defaults are set for sv and ev. </br>
    -n will normalize x coordinates 'window_mean' for all graphs</br>
    -p will allow you to select path for saving files. Default is current path.</br>
    Outputs are plots saved as .png files. Male faimly plots are saved in new directory 'Male Plots/' with each family as plot 
    name. 
    
  ### 3. Kern_Coord.py
  
  What it does:</br>
      This script is optional to run and can be run in any order. </br>
    Takes XML file and plots coordinates of each kernel on plot. Labels kernels whether fluor or nonfluor. Saves plots to new 
    directory with xml file as plot name. </br>
      This script accepts single xml file or directory of xml files as arguments. Outputs scatterplots. </br>
    -p will allow you to select path to save files. Default is current path.

## Modelling Analysis: 

  This group of scripts must be run in this order. Scripts are similar to those above however this generates 5 different models for
  each xml file. At each kernel coordinate, whether the point is fluorescent or nonfluorescent is randomly assigned. 
  
  ### 1. Model.py
  
   This script is nearly identicle to XML_to_ChiSquareTrasmPlot.py</br>
   What it does differently:</br>
      Creates 5 models based on randomly assigning fluor or nonfluor to coordinates. Calculates chi squared statistic for each model.
    Plots positional percent transmission line (colored based on p value) for each model with regression line.</br> 
      This script accepts xml file or directory of xml files, window width, and step size as arguements. Defaults are set
    for width and step size. Optional arguments listed:</br>
    -tk --total_kernels to adjust the minimum number of kernels per entire ear. Default set at 50. Therefore any xml files with 
    less than 50 kernels per ear are skipped.</br>
    -n will normalize the X coordinates "window_mean" for all graphs</br>
    -p --path designates the path to save files. Default is current path.</br>
    Outputs new directory 'Model_Transmission_plots/' with each model graph labeled with file name. Also outputs 2 text files to
    become dataframes in later scripts. 'meta_model_df.txt' contains overall ear statistics for each model. 'everything_model_df.txt'
    contains window calculations and chi square statistics for each ear. Again txt files are tab delimited. 
  
  ### 2. Fam_MODEL_Graphs.py 
  
   This script is nearly identicle to Family_Graphs.py</br>
   What it does differently: </br>
    Creates scatterplots with regression line for each model and overlays them. </br>
     Plots all xml files on single scatterplot with regression line. --5 MODELS PER XML FILE</br>
     Plots all files with male and female in 400s on scatterplot with regression line. --5 MODELS PER XML FILE</br>
     Plots all females in 400s on scatterplot with regression line. --5 MODELS PER XML FILE</br>
     This script accepts 'everything_model_df.txt' as arguments. Outputs are saved graphs. Defaults are set for sv and ev. </br>
     -n will normalize x coordinates 'window_mean' for all graphs</br>
     -p will allow you to select path for saving files. Default is current path.</br>
    
   ### 3. Histogram_Model.py

## Spatial Statistics
Beginning scripts to analyze the spatial distribution of kernels across the ear. 

  ### 1. Spatial_stats.py

## Authors
Elyse Vischulis

## Contributors
Matthew Warrman, Oregon State University
