import os
import glob
import pandas as pd

class CGPS:
    """
    Critical Growth Period Search (CFPS)
    
    Parameters
    ----------
    EnvCorr_path : path
        Data Input Path
    
    """
    def __init__(self, EnvCorr_path):
        self.EnvCorr_path = EnvCorr_path

    def max_corr(self):
        """
        Search for maximum correlation environmental indices 
        and their windows based on critical growth periods derived.
        
        Returns
        ----------
        max_corr_results : pandas.DataFrame
            Return the result with the highest correlation between 
            all environmental factors and all phenotypes
        """
        max_corr_results = []

        def process_file(file_path):
            data = pd.read_csv(file_path)
            results = []
            unique_phenotypes = data['Phen_avg'].unique()
            for phenotype in unique_phenotypes:
                phenotype_data = data[data['Phen_avg'] == phenotype]
                max_corr_row = phenotype_data.loc[phenotype_data['Correlation'].abs().idxmax()]
                results.append(max_corr_row)
            return pd.DataFrame(results)

        for file_path in glob.glob(os.path.join(self.EnvCorr_path, '*.csv')):
            file_name = os.path.basename(file_path).split('.')[0]
            file_results_df = process_file(file_path)
            file_results_df['Env_index'] = file_name  
            max_corr_results.append(file_results_df)

        return pd.concat(max_corr_results, ignore_index=True)

    def EnvAvgCGP(self, dynamic_window_value, max_corr_results):
        """
        Calculation of environmental averages for critical growth periods
        
        Parameters
        ----------
        dynamic_window_value : pandas.DataFrame
            This data is the window mean of the environmental factor calculated by the 
            sliding window function based on the results of the dynamic window function
            Example:
                        Env Leaf_Collar  Threshold   Interval   GDDs_Avg    DLs_Avg    ...
             DEH1_2020          V0        150        1-9        16.693333   14.217213  ... 
             DEH1_2020          VE        216        10-13      18.155000   14.383325  ... 
             DEH1_2020          V1        282        14-16      22.583333   14.459694  ... 
             DEH1_2020          V2        348        17-19      19.321667   14.517497  ...   
             DEH1_2020          V3        414        20-22      23.708333   14.568047  ... 
                ...             ...       ...        ...         ...        ...        ...
 
        max_corr_results ： pandas.DataFrame
            This data is composed of the highest correlation results for each environment
            with each trait phenotype
            Example:
            Phen_avg          Env_avg    Correlation       Env_index
            Pollen_DAP_days   22-22      0.913731          DLs_Avg
            Silk_DAP_days     21-22      0.939654          DLs_Avg
            Plant_Height_cm   6-6        0.509570          DLs_Avg
            Yield_Mg_ha       10-12      0.594732          DLs_Avg
            Grain_Moisture    13-36      -0.806685         DLs_Avg
            
        Returns
        ----------
        results : pandas.DataFrame
            Returns the mean value of the environmental factor in the window with the highest correlation   
        """
        result_frames = []
        leaf_collars = dynamic_window_value['Leaf_Collar'].unique()

        for index, row in max_corr_results.iterrows():
            phenotype = row['Phen_avg']
            env_avg_range = row['Env_avg']
            env_index = row['Env_index']
            range_parts = env_avg_range.split('-')
            start, end = (int(range_parts[0]), int(range_parts[1])) if len(range_parts) == 2 else (int(range_parts[0]), int(range_parts[0]))

            start_leaf = leaf_collars[start - 1] if start - 1 < len(leaf_collars) else None
            end_leaf = leaf_collars[end - 1] if end - 1 < len(leaf_collars) else None
            leaf_collar_range = f'{start_leaf}-{end_leaf}' if start_leaf and end_leaf else None

            if not leaf_collar_range:
                continue

            for env in dynamic_window_value['Env'].unique():
                env_data = dynamic_window_value[(dynamic_window_value['Env'] == env) & (dynamic_window_value['Leaf_Collar'].isin(leaf_collars[start - 1:end]))]
                avg_value = env_data[env_index].mean() if not env_data.empty else None
                current_result = pd.DataFrame({
                    'Env_index': [env_index],
                    'Phenotype': [phenotype],
                    'Environment': [env],
                    'Leaf_Collar_Range': [leaf_collar_range],
                    'Average': [avg_value]
                })
                result_frames.append(current_result)

        return pd.concat(result_frames, ignore_index=True)
