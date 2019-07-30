This repository is a group of scripts and data to analyze an ear of maize. Each script can be run in the terminal and uses argparse as
argument parser.
Scripts and the order they should be run are outlined below:

XML Analysis: 

1. XML_to_ChiSquareTrasmPlot.py 

  What it does:
    Takes xml file, gets X,Y coordinate points for each kernel, and labels as fluorescent or nonfluorescent. 
    Creates sliding window parameter to scan ear counting the number of kernels, fluor, and nonfluor in each window.
    Calculates chi square statistic for entire ear and each individual window.
    Plots positional percent transmission line colored based on whether point is above or below p = 0.05 for chi squared test.
    Also plots and calculates regression.
 
    This script accepts an xml file or directory of xml files, window width, and step size in pixels as arguments.
    Outputs: New directory 'Transmission_plots' containing graphs corresponding to each xml file with positional
    percent transmission for each window, regression, and ear statistics. Also outputs 'meta_df.txt' which contains overall 
    ear stats and p-values. Also outputs 'everything_df.txt' which contains all window calculations and p values for each window for
    each xml file. .txt files are tab delimited use to reconvert to new pandas dataframe in later scripts. 
    
2. Family_Graphs.py

  What it does: 
    Takes 'everything_df.txt' and creates a variety of plots based on male/female crosses. 
    Plots all xml files on single scatterplot with regression line. 
    Plots all files with male and female in 400s on scatterplot with regression line.
    Plots all females in 400s on scatterplot with regression line.
    Plots each male family in 400s with one or more lines on each graph corresponding to each xml file in family. Also plots
    regression lines.
    
    This script accepts everything_df.txt, starting value for male family plots (must be in 400s), and ending value for male
    family plots (must be in 400s). Example) 410 499
    Outputs are plots saved as .png files. Male faimly plots are saved in new directory 'Male Plots/' with each family as plot 
    name. 
    
  3. Kern_Coord.py
  
    What it does:
    This script is optional to run and can be run in any order. 
    Takes XML file and plots coordinates of each kernel on plot. Labels kernels whether fluor or nonfluor. Saves plots to new 
    directory with xml file as plot name. 
    
    This script accepts single xml file or directory of xml files as arguments. Outputs scatterplots. 
    
Modelling Analysis: 

  This group of scripts must be run in this order. Scripts are similar to those above however this generates 5 different models for
  each xml file. At each kernel coordinate, whether the point is fluorescent or nonfluorescent is randomly assigned. 
  
  1. Model.py
  
    This script is nearly identicle to XML_to_ChiSquareTrasmPlot.py
    What it does differently:
    Creates 5 models based on randomly assigning fluor or nonfluor to coordinates. Calculates chi squared statistic for each model.
    Plots positional percent transmission line (colored based on p value) for each model with regression line. 
    
    This script accepts xml file or directory of xml files, window width, and step size as arguements.
    Outputs new directory 'Model_Transmission_plots/' with each model graph labeled with file name. Also outputs 2 text files to
    become dataframes in later scripts. 'meta_model_df.txt' contains overall ear statistics for each model. 'everything_model_df.txt'
    contains window calculations and chi square statistics for each ear. Again txt files are tab delimited. 
  
  2. Fam_MODEL_Graphs.py 
  
    This script is nearly identicle to Family_Graphs.py
    What it does differently: 
    Creates scatterplots with regression line for each model and overlays them. 
    Plots all xml files on single scatterplot with regression line. --5 MODELS PER XML FILE
    Plots all files with male and female in 400s on scatterplot with regression line. --5 MODELS PER XML FILE
    Plots all females in 400s on scatterplot with regression line. --5 MODELS PER XML FILE
    
    This script accepts 'everything_model_df.txt' as arguments. Outputs are saved graphs. 
    
   3. Histogram_Model.py
   
Spatial Statistics: 
  1. Spatial_stats.py
    
