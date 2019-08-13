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
    -a will adjust chi square p values according to Benjamini–Hochberg procedure with statsmodels. This takes into account multiple comparisons, however, it is not the best choice for a test. This should be changed in the future. Statsmodels has a variety of other tests that could easily be changed. You can read more here: </br>
    https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html </br>
    Outputs: New directory 'Transmission_plots' containing graphs corresponding to each xml file with positional
    percent transmission for each window, regression, and ear statistics. Also outputs 'meta_df.txt' which contains overall 
    ear stats and p-values. Also outputs 'everything_df.txt' which contains all window calculations and p values for each window for
    each xml file. .txt files are tab delimited use to reconvert to new pandas dataframe in later scripts. </br>
![X400x417-3m1](https://user-images.githubusercontent.com/52712211/62904778-83d4b500-bd1c-11e9-9ea6-9d0390a18d04.png)

    
### 2. Family_Graphs.py

  What it does:</br>
      Takes 'everything_df.txt' and creates a variety of plots based on male/female crosses. </br>
    Plots all xml files on single scatterplot with regression line. </br>
    Plots all files with male and female in 400s on scatterplot with regression line.</br>
    Plots all females in 400s on scatterplot with regression line.</br>
    Plots each male family in 400s with one or more lines on each graph corresponding to each xml file in family. Also plots
    regression lines.</br>
    Plots regression line from each male family colored based on expression. </br>
    Plots regression line from each male family colored based on known transmission defect or none. </br>
      This script accepts everything_df.txt, starting value for male family plots (must be in 400s), and ending value for male
    family plots (must be in 400s). Example) 410 499. Defaults are set for sv and ev. Also requires Table8.tsv from Data_for_Analysis folder. </br>
    -n will normalize x coordinates 'window_mean' for all graphs</br>
    -p will allow you to select path for saving files. Default is current path.</br>
    Outputs are plots saved as .png files. Male faimly plots are saved in new directory 'Male Plots/' with each family as plot 
    name. </br>
   ![everything_norm_plot](https://user-images.githubusercontent.com/52712211/62904771-833c1e80-bd1c-11e9-87cf-fed5683508c5.png)</br>
   ![Everything_norm_400](https://user-images.githubusercontent.com/52712211/62904769-833c1e80-bd1c-11e9-985b-fc12b9964479.png)</br>
   ![Female 400s Norm Cross Plot](https://user-images.githubusercontent.com/52712211/62904773-833c1e80-bd1c-11e9-9700-146806df4a6d.png)</br>
   ![417](https://user-images.githubusercontent.com/52712211/62904766-82a38800-bd1c-11e9-88e4-6e9111653680.png)</br>
   ![Exp_Reg_Plot](https://user-images.githubusercontent.com/52712211/62904772-833c1e80-bd1c-11e9-8800-f976b2d1d5bc.png)</br>
   ![Trans_Reg_Plot](https://user-images.githubusercontent.com/52712211/62904776-83d4b500-bd1c-11e9-8ec5-6cfc0f6de0ff.png)
    
  ### 3. Kern_Coord.py
  
  What it does:</br>
      This script is optional to run and can be run in any order. </br>
    Takes XML file and plots coordinates of each kernel on plot. Labels kernels whether fluor or nonfluor. Saves plots to new 
    directory with xml file as plot name. </br>
      This script accepts single xml file or directory of xml files as arguments. Outputs scatterplots. </br>
    -p will allow you to select path to save files. Default is current path.</br>
![X401x492-2m1](https://user-images.githubusercontent.com/52712211/62904779-846d4b80-bd1c-11e9-847e-9a63a20da60b.png)</br>
### 4. Male_plot_folder.py
This script allows you to 'drag and drop' desired xml files into a folder, process all files, and output a male family plot based on everything in the folder. Input arguments are xml file or directory of xml files. Ouptut is one male family plot. It is really important that you designate the path where you want it saved, as all graphs have the same title. Graphs can be distingushed apart by the key but that may be annoying. </br>
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
    -a will adjust for multiple comparisons of the p-values from chi square test using Benjamini–Hochberg procedure. </br>
    Outputs new directory 'Model_Transmission_plots/' with each model graph labeled with file name. Also outputs 2 text files to
    become dataframes in later scripts. 'meta_model_df.txt' contains overall ear statistics for each model. 'everything_model_df.txt'
    contains window calculations and chi square statistics for each ear. Again txt files are tab delimited. </br>
  ![X400x417-3m1_model](https://user-images.githubusercontent.com/52712211/62904777-83d4b500-bd1c-11e9-832c-965f8eb482ea.png)

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
![everything_norm_MODEL_plot](https://user-images.githubusercontent.com/52712211/62904770-833c1e80-bd1c-11e9-82e5-929c1fc7263a.png)</br>
![Everything_400_norm_MODEL](https://user-images.githubusercontent.com/52712211/62904768-833c1e80-bd1c-11e9-9d47-21f8543d6e5e.png)</br>
![Female MODEL norm 400s Cross Plot](https://user-images.githubusercontent.com/52712211/62904774-83d4b500-bd1c-11e9-887e-a22c80a5d925.png)</br>
## Histograms
   ### 1. Histogram.py
   From model_meta_df.txt or model_normalized_meta_df.txt, and meta_df.txt or meta_normalized_df.txt creates 6 histograms (R-squared, slope, and p-value) based on regression statistics for all models and all normal xmls. </br>
  This script accepts the model_meta_df.txt and meta_df.txt as an argument. Optional arguments are '-p' to determine path where plots are saved. Default is current path. If -f will only make histograms of X4..x4.. crosses in file names. If -s will only make histogram for the 492 male family. </br>
Example Histograms:</br>
![Model_400sSlope_Hist](https://user-images.githubusercontent.com/52712211/62904775-83d4b500-bd1c-11e9-9e5d-0dc096dd4eb3.png)</br>
![XML_400sSlope_Hist](https://user-images.githubusercontent.com/52712211/62904780-846d4b80-bd1c-11e9-97cf-7d857032991b.png)</br>

### 2. Histogram_defect.py
From model_meta_df.txt and meta_df.txt, will make histograms for transmission defect or no transmission defect for both normal xml files and model. Requires Table8.tsv as argument as well. Can designate new path for saving plots with -p otherwise current path is default.</br>
![Defect_Slope_Hist](https://user-images.githubusercontent.com/52712211/62905279-994ade80-bd1e-11e9-9281-05bb2b7cbf75.png)</br>
![Model_defect_Slope_Hist](https://user-images.githubusercontent.com/52712211/62905280-994ade80-bd1e-11e9-9791-fab8fff46dd9.png)</br>
![Model_nodefect_Slope_Hist](https://user-images.githubusercontent.com/52712211/62905282-994ade80-bd1e-11e9-97b1-c8345211bea1.png)</br>
![No_defect_Slope_Hist](https://user-images.githubusercontent.com/52712211/62905283-994ade80-bd1e-11e9-8558-6c6c02c91979.png)</br>

## Spatial Statistics
Beginning scripts to analyze the spatial distribution of kernels across the ear. 

  ### 1. Spatial_stats.py
 This is very beginning steps to begin statstically evaluating the spatial distribution of kernel coordinates. Accepts xml file, or directory of xml files as arguments. Then computes quadrat based statistics for homogeneous planar points. Output is a giant dataframe of p-values and pseudo-pvalues for evaluation. I believe our data must be evaluated based on inhomogeneous poisson processes so new stats must be applied, perhaps using R spatstat package. More info for what is involved in this script can be found here:</br> https://nbviewer.jupyter.org/github/pysal/pointpats/blob/master/notebooks/Quadrat_statistics.ipynb </br>
 https://nbviewer.jupyter.org/github/pysal/pointpats/blob/master/notebooks/distance_statistics.ipynb </br>
 ### 2. Export_xml_coord.py
 This script will take a single xml, or directory of xml files, as an arugument and output kernel coordinates to a tab-delimited text file. This could potentially be used for R spatstat. 
## Authors
Elyse Vischulis

## Contributors
Matthew Warman, Oregon State University
