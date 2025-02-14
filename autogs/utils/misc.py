from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from matplotlib import pyplot as plt 
import pandas as pd 
import numpy as np

def outlier_detector(df, column_key_start, column_key_end, pca_comp = 0.80, neighbours = 100, plot_PCA = False):
    """
    Function using SKlearn's 'LocalOutlierFactor' to detec outliers within a specififed range of the input df's columns
    
    Input:
        df: dataframe from which to select sub-dataframe
        column_key_start: name of column to start 
        column_key_end: nume of column end including column itself
        pca_comp: number of Principal components to include (int for number, float for fraction explained variance)
        neighbours: number of neighbours to conisder in localOutlier
        plot_PCA: if true will plot first two principal componenets and colour outliers
    """
    

    # -- Select specified columns
    idx_measurements = list(df.keys()).index(column_key_start)
    idx_measurements_end = list(df.keys()).index(column_key_end) + 1  # plus 1 because [5:10] does not include idx 10
    data = df.iloc[:, idx_measurements : idx_measurements_end]
    
    # -- change processing depending on whether PCA should be invoked or not
    if (type(pca_comp) == float) | (type(pca_comp) == int) :

        # -- apply standard scaler (outliers will remain)
        x = StandardScaler().fit_transform(data)
        
        # -- select fraction or number of Principal componenets and create PCA
        pca = PCA(n_components = pca_comp)
        
        # -- apply PCA
        X = pca.fit_transform(x)
        
    else:
        X = data
        
    # -- create outlier detector
    outlier_detector = LocalOutlierFactor(n_neighbors=neighbours)
    
    # -- apply detector on data
    inliers = outlier_detector.fit_predict(X)
    
    # -- create df with inliers only
    df_outliers_removed = df[inliers==1] # inliers = 1
    
    if plot_PCA == True:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c = inliers)
        plt.xlabel(r"Principal Component 1"); plt.ylabel(r"Principal Component 2")
        plt.title("Outliers")
        plt.show()
        
    return df_outliers_removed



def envelope(df, param_x, param_y, begin, end, steps =25, log = True):
    """
    function to derive the median and quantiles for a pointcloud from a df with two specified parameters
    """
    
    placeholder = df.copy()
    
    if log == True:
        bins = np.logspace(begin, end, steps)
    else:
        bins=np.linspace(begin, end, steps)
        
    placeholder['bins_x'], bins = pd.cut(abs(placeholder[param_x]), bins=bins, include_lowest=True, retbins=True)
    placeholder['bins_y'], bins = pd.cut(abs(placeholder[param_y]), bins=bins, include_lowest=True, retbins=True)
        
    bin_center = (bins[:-1] + bins[1:]) /2
    bin_median = abs(placeholder.groupby('bins_x')[param_y].agg(np.nanmedian))#.nanmedian())
    bin_count_x = abs(placeholder.groupby('bins_x')[param_x].count())
    bin_count_y = abs(placeholder.groupby('bins_y')[param_y].count())
    bin_std = abs(placeholder.groupby('bins_x')[param_y].agg(np.nanstd)) #.nanstd())
    bin_quantile_a = abs(placeholder.groupby('bins_x')[param_y].agg(lambda x: np.nanpercentile(x, q = 2.5)))
    bin_quantile_b = abs(placeholder.groupby('bins_x')[param_y].agg(lambda x: np.nanpercentile(x, q = 16)))
    bin_quantile_c = abs(placeholder.groupby('bins_x')[param_y].agg(lambda x: np.nanpercentile(x, q = 84)))
    bin_quantile_d = abs(placeholder.groupby('bins_x')[param_y].agg(lambda x: np.nanpercentile(x, q = 97.5)))
    return bin_center, bin_median, bin_count_x, bin_count_y, bin_std, bin_quantile_a, bin_quantile_b, bin_quantile_c, bin_quantile_d

def plot_envelope_single(df_plot, param_test, param_predict, hist_steps, title, x_axis_title, y_axis_title, alpha = 1, legend = True, axis_scale = 'log', ax_min = 0.5, ax_max = 10000 ):

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,figsize=(7,6))
    
    fontsize = 15
    ax2 = axes
    ax2_2 = ax2.twinx()
    bin_center, bin_median, bin_count_test, bin_count_pred, bin_std, bin_quantile_a, bin_quantile_b, bin_quantile_c, bin_quantile_d = envelope(df_plot, param_test, param_predict, \
                                                                                                                             -1, 4, steps =hist_steps, log = axis_scale == True)
    ax2.plot(bin_center, bin_median, 'k', label = r'Median')
    ax2.fill_between(bin_center, bin_quantile_b, bin_quantile_c, color = 'gray', alpha = 0.8, label = r'$68\%%$')
    ax2.fill_between(bin_center, bin_quantile_a, bin_quantile_d, color = 'gray', alpha = 0.6, label = r'$95\%%$')
    
    ax2.set_xscale(axis_scale)
    ax2.set_yscale(axis_scale)
    ax2.plot([ax_min*2, ax_max/2], [ax_min*2, ax_max/2], '--k')
    ax2.set_ylim(ax_min,ax_max)
    ax2.set_xlim(ax_min,ax_max)
    # ax2.legend(fontsize = fontsize)
    # ax2.set_title('Obukhov length prediction (Test)', fontsize=fontsize)
    ax2.set_ylabel(y_axis_title, fontsize=fontsize)
    ax2.set_xlabel(x_axis_title, fontsize=fontsize)
    
    ax2.tick_params(which='major', width=3, length=6)
    ax2.tick_params(which='minor', width=1.5, length=3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    
    # ax2_2.bar(bin_center[1:], bin_count_pred.values[1:], width= np.diff(bin_center), hatch="X", edgecolor = 'k', color = 'gray', alpha = 0.5)
    ax2_2.step(bin_center[1:], bin_count_pred.values[1:], where = 'mid', color = 'r', alpha = 0.5, linewidth = 3)
    ax2_2.step(bin_center[1:], bin_count_test.values[1:], where = 'mid', color = 'b', alpha = 0.5, linewidth = 3)
    # ax2_2.bar(bin_center[1:], bin_count_test.values[1:], width= np.diff(bin_center), color = 'none', edgecolor = 'k')
    ax2_2.tick_params(axis='y', colors='b')
    # ax2_2.set_yticks([0, 0.05, 0.10, 0.15, 0.2])
    # ax2_2.set_xscale(axis_scale)
    # ylim = 0.20# int(bin_count_test.max()*4)
    # ax2_2.set_ylim(0,ylim)
    # ax2_2.set_ylabel('Relative freq.', color='b', fontsize=fontsize)
    ax2_2.set_xscale('log')
    ylim = int(bin_count_test.max()*4)
    ax2_2.set_ylim(0,ylim)
    ax2_2.set_ylabel('Hist. count', color='b', fontsize=fontsize)
    
    # -- plot custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    if legend == True:
        legend_elements = [Line2D([0], [0], color='k', lw=4, label = 'Median'),
                           Patch(facecolor='gray', alpha = 0.8, edgecolor='none', label = '68%'),
                           Patch(facecolor='gray', alpha = 0.6, edgecolor='none', label = '95%'),
                           # Patch(hatch="X", edgecolor = 'k', facecolor = 'none', label='Val.'),
                           Line2D([0], [0], color='r', alpha = 0.5, linewidth = 3, label = 'Est.'),
                           Line2D([0], [0], color='b', alpha = 0.5, linewidth = 3, label = 'Val.')]
        plt.legend(handles = legend_elements, framealpha =0.99, edgecolor = 'black', borderpad = 0.2, 
                   loc = 'upper left', ncol = 2, fontsize = fontsize)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    # plt.subplots_adjust(wspace=0.3)
    # plt.show()
    
    return bin_center, bin_median, bin_count_test, bin_count_pred, bin_std, fig