import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.colors import LinearSegmentedColormap, Normalize

def prepare_data(env_df, phenotype_df, env_index, phenotype):
    """筛选环境变量和表型数据，去除缺失值。"""
    X = env_df.query("Env_index == @env_index and Phenotype == @phenotype")['Average'].tolist()
    Y_numeric = phenotype_df.select_dtypes(include=[np.number])
    Y = np.array(Y_numeric.T)
    
    mask = ~np.isnan(Y).any(axis=0)
    X = np.array(X)[mask]
    Y = Y[:, mask]
    
    return X, Y, Y_numeric.columns

def compute_regression(X, Y):
    """计算每个变量的线性回归斜率和截距。"""
    results = np.apply_along_axis(lambda y: linregress(X, y)[:2], axis=1, arr=Y)
    return results[:, 0], results[:, 1]

def plot_regression(X, Y, slopes, intercepts, env_names, env_index, phenotype, save_path=None):
    """绘制回归直线和散点图。"""
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#96fcf9', '#8532ff'])
    norm = Normalize(vmin=min(slopes), vmax=max(slopes))
    
    plt.figure(figsize=(7, 3))
    x_min, x_max = min(X), max(X)
    x_range = x_max - x_min
    x_buffer = x_range * 0.02
    extended_X = np.linspace(x_min - x_buffer, x_max + x_buffer, 500)
    
    for i in range(Y.shape[0]):
        plt.plot(extended_X, slopes[i] * extended_X + intercepts[i], 
                 color=cmap(norm(slopes[i])), linewidth=0.2, zorder=1)
    
    for j in range(Y.shape[1]):
        plt.scatter([X[j]] * len(Y[:, j]), Y[:, j], color='#8532ff', s=0.1, zorder=2)
        plt.text(X[j], np.mean(Y[:, j]), env_names[j], fontsize=6, verticalalignment='bottom')
    
    plt.xlabel(env_index)
    plt.ylabel(phenotype)
    plt.xlim(x_min - x_buffer, x_max + x_buffer)
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.show()

def save_regression_results(columns, slopes, intercepts, output_csv):
    """保存回归系数到 CSV。"""
    df = pd.DataFrame({'Hybrid': columns, 'Slope': slopes, 'Intercept': intercepts})
    df.to_csv(output_csv, index=False)

def slope_inter(env_df, phenotype_df, env_index, phenotype, result_path=None):
    """主函数，执行数据处理、回归分析和可视化。"""
    X, Y, hybrid_names = prepare_data(env_df, phenotype_df, env_index, phenotype)
    slopes, intercepts = compute_regression(X, Y)
    env_names = phenotype_df['Env'].iloc[:len(X)]
    
    if result_path:
        base_path = f"{result_path}_{env_index}_{phenotype}"
        plot_regression(X, Y, slopes, intercepts, env_names, env_index, phenotype,save_path=f'{base_path}_ha.pdf')
        save_regression_results(hybrid_names, slopes, intercepts, output_csv=f'{base_path}_KB.csv')

    return slopes, intercepts 