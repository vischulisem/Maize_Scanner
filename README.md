# Overview of Maize Scanner :corn:
This repository is a group of scripts and data to analyze an ear of maize. Each script can be cloned and run in the terminal and uses argparse as argument parser.</br>
Scripts and the order they should be run are outlined below.

## Prerequisites
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
![X400x417-3m1](https://user-images.githubusercontent.com/52712211/62398816-a4d42380-b52e-11e9-9f3b-232c5fcbc4ca.png)

    
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
    name. </br>
    ![everything_norm_plot](https://user-images.githubusercontent.com/52712211/62398922-fed4e900-b52e-11e9-94cb-5870abe7e0c8.png)</br>
![Everything_norm_400](https://user-images.githubusercontent.com/52712211/62398933-03999d00-b52f-11e9-913e-b90652330639.png)</br>
![Female 400s Norm Cross Plot](https://user-images.githubusercontent.com/52712211/62398938-07c5ba80-b52f-11e9-9a84-bdbb62dd053c.png)</br>
![491](https://user-images.githubusercontent.com/52712211/62398948-0e543200-b52f-11e9-9524-1a754cf5b21c.png)</br>
    
  ### 3. Kern_Coord.py
  
  What it does:</br>
      This script is optional to run and can be run in any order. </br>
    Takes XML file and plots coordinates of each kernel on plot. Labels kernels whether fluor or nonfluor. Saves plots to new 
    directory with xml file as plot name. </br>
      This script accepts single xml file or directory of xml files as arguments. Outputs scatterplots. </br>
    -p will allow you to select path to save files. Default is current path.

## Modelling Analysis

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
  
  ![X400x417-3m1_model](https://user-images.githubusercontent.com/52712211/62398878-e238b100-b52e-11e9-8f41-a4b7d994a578.png)  
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
    ![everything_norm_MODEL_plot](https://user-images.githubusercontent.com/52712211/62399016-48253880-b52f-11e9-910b-e1448c4498e4.png)</br>
![Everything_400_norm_MODEL](https://user-images.githubusercontent.com/52712211/62399023-4bb8bf80-b52f-11e9-95ef-0c758df6771d.png)</br>
![Female MODEL norm 400s Cross Plot](https://user-images.githubusercontent.com/52712211/62399030-4fe4dd00-b52f-11e9-92e5-8e2bd8f177aa.png)</br>

   ### 3. Histogram_Model.py
   From model_meta_df.txt or model_normalized_meta_df.txt, creates 3 histograms (R-squared, slope, and p-value) based on regression statistics for all models. </br>
  This script accepts the model_meta_df.txt as an argument. Optional arguments are '-p' to determine path where plots are saved. Default is current path. </br>
![Model_Pvalue_Hist](https://user-images.githubusercontent.com/52712211/62399071-6c811500-b52f-11e9-8b00-28968ce9b77f.png)</br>
![Model_Rsq_Hist](https://user-images.githubusercontent.com/52712211/62399072-6c811500-b52f-11e9-81b0-bb64aee5350d.png)</br>
![Model_Slope_Hist](https://user-images.githubusercontent.com/52712211/62399073-6c811500-b52f-11e9-9888-f4a8ee2389d1.png)</br>
## Spatial Statistics
Beginning scripts to analyze the spatial distribution of kernels across the ear. 

  ### 1. Spatial_stats.py
 

## Authors
Elyse Vischulis

## Contributors
Matthew Warrman, Oregon State University
