import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

def GxE_Plot(data, value_column,save_file):
    """
    Plot to visualize the phenotypic plasticity in maize across different environments and hybrids.
    
    Parameters:
    - data: A Pandas DataFrame containing environment (Env), hybrid (Hybrid), and value columns.
    - value_column: The name of the column to use for plotting values.
    """
    # Define the color list for gradients and the output filename
    color_list = ['#8E4CBA', 'cyan']
    mean_point_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    #save_file = f'{value_column}_GxE.pdf'
    
    # Ensure there are no NaN values in the specified columns in the data
    data.dropna(subset=['Env', value_column], inplace=True)

    # Normalize the specified 'value_column' within each environment
    data['normalized_value'] = data.groupby('Env')[value_column].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Create a color mapping using the defined color list
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', color_list)

    plt.figure(figsize=(8, 5))
    marker_size = 8

    # Plot each hybrid's data as scatter points with gradient colors and connect them with lines
    for hybrid in data['Hybrid'].unique():
        subset = data[data['Hybrid'] == hybrid]
        colors = custom_cmap(subset['normalized_value'])
        
        plt.scatter(subset['Env'], subset[value_column], c=colors, label=hybrid, s=marker_size)
        # Ensure the correct order of 'Env' when connecting points with lines
        sorted_subset = subset.sort_values(by='Env')
        for i in range(len(sorted_subset['Env']) - 1):
            plt.plot(sorted_subset['Env'].iloc[i:i+2], sorted_subset[value_column].iloc[i:i+2], c=colors[i], linewidth=0.2)

    # Calculate and plot the mean values for each environment
    mean_values = data.groupby('Env')[value_column].mean().reset_index().sort_values(by='Env')
    plt.plot(mean_values['Env'], mean_values[value_column], color='black', linestyle='-', linewidth=0.8)
    # Plot mean value points as hollow circles with different colors and add text labels for mean values
    for i, row in mean_values.iterrows():
        plt.scatter(row['Env'], row[value_column], s=50, facecolors='none', 
                    edgecolors=mean_point_colors[i % len(mean_point_colors)], linewidths=1.5, zorder=3)
        plt.text(row['Env'], row[value_column] + 5, f"{row[value_column]:.2f}", 
                 color=mean_point_colors[i % len(mean_point_colors)], ha='center', va='bottom')
        
    plt.xticks(rotation=0)  # Rotate the 'Env' axis labels to be vertical

    plt.ylabel(value_column)
    plt.savefig(save_file+value_column+'_GxE.pdf', bbox_inches='tight')  # Save the figure to a file
    plt.grid(False)  # Disable the grid for a cleaner plot
    #plt.show()  # Display the plot