# User can input low and high parameter corresponding to number of 400 male family to plot
# Files are sorted and grouped together, x coordinates (window mean) normalized for all lines
def male_fam_plot(input_df, low, high, path):
    print(f'Starting male plots...')
    big_reg_df = pd.DataFrame(columns='Male_Fam Normalized_Window_Mean Percent_Transmission'.split())
    for i in range(low, high):
        # Reading in txt file as dataframe
        data = pd.read_csv(input_df, sep="\t")
        # Sorting through file names while iterating through range
        data = data[data['File'].str.contains(r'x4..')]
        search_values = ['x' + str(i)]
        data = data[data.File.str.contains('|'.join(search_values))]

        # If family doesn't exist as filename, go to next i
        if data.empty:
            continue
        else:
            # Group all filenames together and normalize window mean x coordinates
            data['Normalized_Window_Mean'] = data.groupby('File')['window_mean'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            # Plotting begins
            sns.set_style("white")
            male = sns.lineplot(x="Normalized_Window_Mean", y="Percent_Transmission", data=data, hue="File", linewidth=5)
            sns.set(rc={'figure.figsize': (11.7, 8.27)})
            plt.ylim(0, 1)
            male.set(yticks=[0, 0.25, 0.5, 0.75, 1])
            male.yaxis.grid(True)
            male.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
            plt.title(repr(i) + ' Plot', fontsize=30, weight='bold', loc='center', verticalalignment='baseline')
            plt.xlabel('Normalized Window Position (pixels)', fontsize=18, weight='bold')
            plt.ylabel('Percent Transmission', fontsize=18, weight='bold')
            # Regression calculations
            reg_x = data['Normalized_Window_Mean'].values
            reg_y = data['Percent_Transmission'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
            r2 = r_value ** 2

            reg_xx = data['Normalized_Window_Mean'].values.tolist()
            reg_yy = data['Percent_Transmission'].values.tolist()
            llist = [str(i)] * len(reg_xx)
            reg_list = [list(a) for a in zip(llist, reg_xx, reg_yy)]
            reg_df = pd.DataFrame(data=reg_list, columns='Male_Fam Normalized_Window_Mean Percent_Transmission'.split())
            big_reg_df = big_reg_df.append(reg_df)
            big_reg_df = big_reg_df.reset_index(drop=True)

            # Plotting regression line
            plt.plot(reg_x, intercept + slope * reg_x, 'r', label='fitted line', color='red', linewidth=3,
                     dashes=[5, 3])

            # Creating empty df so that mean rsquared for each line can be displayed
            stat_df = pd.DataFrame(columns='Slope Intercept RSquared P-Value'.split())
            # Iterating through each file name and plotting regression line
            grps = data.groupby(['File'])
            for file, grp in grps:
                iter_y = grp['Percent_Transmission']
                iter_x = grp['Normalized_Window_Mean']
                slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(iter_x, iter_y)
                plt.plot(iter_x, intercept2 + slope2 * iter_x, 'r', label='fitted line', color='black', linewidth=3,
                         dashes=[5, 3])
                # Putting regression stats for each file into dataframe
                this_stat = [[slope2, intercept2, r_value2 ** 2, p_value2]]
                temp_stat_df = pd.DataFrame(data=this_stat, columns='Slope Intercept RSquared P-Value'.split())
                stat_df = stat_df.append(temp_stat_df)
                stat_df = stat_df.reset_index(drop=True)

            # Calculating mean values for each stat and rounding to 4 decimal places
            average_slope = round(stat_df['Slope'].mean(), 4)
            average_intercept = round(stat_df['Intercept'].mean(), 4)
            average_Rsquared = round(stat_df['RSquared'].mean(), 4)
            average_Pval = '{:0.3e}'.format(stat_df['P-Value'].mean())
            combined_pval = '{:0.3e}'.format(p_value)
            # Text string for text box
            textstr = '\n'.join((f'Combined Slope = {round(slope, 4)}',
                                 f'Average Indv. Slope = {average_slope}',
                                 f'Combined Intercept = {round(intercept, 4)}',
                                 f'Average Indv. Intercept = {average_intercept}',
                                 f'Combined P-value = {combined_pval}',
                                 f'Average Indv. P-value = {average_Pval}',
                                 f'Combined R-Squared = {round(r_value ** 2, 4)}',
                                 f'Average Indv. R-squared = {average_Rsquared}'))
            # Creating text box on graph
            male.text(0.05, 0.95, textstr, transform=male.transAxes, fontsize=10, fontweight='bold',
                    verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10, 'edgecolor': 'black'})

            # Saving figure in new directory with cross file name
            male_graph = male.get_figure()

            # create directory to save plots
            script_dir = path
            results_dir = os.path.join(script_dir, 'Output_Family_Graphs/Male_Plots/')
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            # sample_file_name
            sample_file_name = repr(i) + '.png'

            male_graph.savefig(results_dir + sample_file_name, bbox_inches="tight")
            plt.close()
            print(f'{i} Plot Completed.')
    print(f'Done male plots!')
    big_reg_df['Male_Fam'] = big_reg_df['Male_Fam'].astype(np.int64)
    return big_reg_df
