import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Sliding Window Algorithm
class WindCal:
    """
    A is defined to compute the environmental window mean. We provide two types of calculations,
    a sliding window calculation with a fixed window size, and a dynamic window calculation with
    a variable window size.
    
    '''
    Parameters
    ----------
    data : pandas.DataFrame
    Example of input data
    #   Env         Date         GDDs
    #   DEH1_2020   2020-05-14   6.100
    #   DEH1_2020   2020-05-15   12.715
    #   DEH1_2020   2020-05-16   10.895
    #   DEH1_2020   2020-05-17   6.205
    #   DEH1_2020   2020-05-18   5.165
    #   ...         ...          ...
    """

    def __init__(self, data):
        self.data = data
    
    @staticmethod
    def sliding_window(data, k, n):
        """
        This function is utilized to calculate the combination of time windows within the reproductive cycle
        
        Parameters
        ----------
        k : int
            The size of the sliding window
        n : int
            Size of the crop growth periods

        Returns
        ----------
        window_average : float
            Average of environmental factors across all window combinations
        """
        # The size of the window cannot exceed the size of the fertility cycle
        if k > n:
            raise ValueError("The size of the window (k) cannot be larger than the fertility cycle (n)")

        # Extract the environment name from the input file
        unique_environments = data['Env'].unique() 
        result_dict = {}
        window_label_dict = {}  

        for environment in unique_environments:
            environment_data = data[data['Env'] == environment]
            environment_data = np.array(environment_data.iloc[:,2:])
            results = []
            window_labels = []  

            for window_size in range(k, n + 1):
                window_sum = sum(environment_data[:window_size])
                window_start = 1
                window_end = window_size

                while window_end <= n:
                    window_average = window_sum / window_size
                    window_label = f'{window_start}-{window_end}'

                    # Append window_label and window_average as a list
                    results.append(window_average)
                    window_labels.append(window_label) 

                    if window_end == len(environment_data):
                        break

                    window_sum -= environment_data[window_start - 1]
                    window_sum += environment_data[window_end]
                    window_start += 1
                    window_end += 1

            window_label_dict[environment] = window_labels

            # Create DataFrame from the results with proper column labels
            df = pd.DataFrame(results, columns=[environment])
            result_dict[environment] = df

        # Combine results for multiple environments into a single DataFrame
        final_df = pd.concat(result_dict.values(), axis=1, keys=result_dict.keys())
        window_label_df = pd.DataFrame(window_label_dict)
        final_df = final_df.groupby(level=0, axis=1).mean()
        window_label_df.columns = ['win_label'] * len(window_label_df.columns)

        final_df = pd.concat([window_label_df, final_df], axis=1)
        window_average  = final_df.loc[:,~final_df.columns.duplicated()]

        return window_average

    def dynamic_window(self, ref_wind):
        """
        Define this function for calculating the window of explicit reproductive period, which needs/
        to refer to the cumulative temperature accumulation of the corn and is therefore dynamic

        Parameters
        ----------
        ref_wind ： pandas.DataFrame
            Critical growth periods for maize
            #  Leaf_Collar  GDUs_Stage  GDUs_after_seeding
            #  V0           150         150
            #  VE           66          216
            #  ...          ...         ...

        Returns
        ----------
        dynamic_window_average : float,pandas.DataFrame
            Environmental averages for windows of critical growth periods

        """

        dynamic_window_average = []

        # Identifying environmental indices columns, assuming 'Env' and 'Date' are not environmental indices
        env_indexs = [col for col in self.data.columns if col not in ['Env', 'Date']]

        # Resetting index for each new environment
        self.data = self.data.reset_index(drop=True)

        last_index = -1 # Initialize the index of the previous threshold
        cumulative_sum = 0 # Initialize the cumulative sum

        # Mapping GDUs_after_seeding to Leaf_Collar stages (using the correct column names)
        gdu_to_leaf_collar = dict(zip(ref_wind['GDUs_after_seeding'], ref_wind['Leaf_Collar ']))

        for threshold in ref_wind['GDUs_after_seeding']:
            cum_sum_from_last_index = self.data['GDDs'][last_index + 1:].cumsum()
            threshold_indices = (cum_sum_from_last_index + cumulative_sum >= threshold)

            if not threshold_indices.any():
                continue     
            else:
                threshold_index = threshold_indices.idxmax()

            interval_data = self.data.iloc[last_index + 1:threshold_index + 1]
            interval_averages = {f'{env_index}_Avg': interval_data[env_index].mean() for env_index in env_indexs}

            dynamic_window_average.append({
                'Env': self.data['Env'].iloc[0],
                'Leaf_Collar': gdu_to_leaf_collar.get(threshold, 'Unknown'),
                'Threshold': threshold,
                'Interval': f"{last_index + 2}-{threshold_index + 1}",
                **interval_averages
            })

            cumulative_sum += interval_data['GDDs'].sum()
            last_index = threshold_index

        return pd.DataFrame(dynamic_window_average)

class CorrCal:
    """
    CorrelationCalculator is defined to calculate the coefficient of correlation between phenotypic
    means and environmental factor means in multiple environments. We provide two
    types of calculations, a sliding window calculation with a fixed window size
    and a dynamic window calculation with an indeterminate window size.
    
    '''
    Parameters
    ----------
    phen : pandas.DataFrame
        Example of input phenotype data
        
        Env         Pollen_DAP_days   Silk_DAP_days  ASI         ...
        DEH1_2020   59.899181         59.982185      0.100501    ...
        DEH1_2021   69.507772         69.848521      0.347142    ...
        IAH2_2021   NaN               NaN            NaN         ...
        IAH3_2021   NaN               NaN            NaN         ...
        IAH4_2021   75.322229         76.238425      0.903884    ...
        WIH2_2020   66.813311         68.485909      1.718739    ...
        WIH2_2021   64.407103         64.687778      0.304570    ...
        ...         ...               ...            ...         ...
   
    env ： pandas.DataFrame
        We provide two types of data, the first is the mean value of the environmental
        factors calculated in a fixed window; the second is the mean value of the environmental
        factors calculated in a dynamic window. The second can be used for all species, and the
        first is a modified version for maize; please refer to the paper for a detailed description
        of the data.
        
        Env           Leaf_Collar   Threshold   Interval   GDDs_Avg    ...
        DEH1_2020     V0            150         1-9        16.693333   ... 
        DEH1_2020     VE            216         10-13      18.155000   ...
        DEH1_2020     V1            282         14-16      22.583333   ...
        DEH1_2020     V2            348         17-19      19.321667   ...
        DEH1_2020     V3            414         20-22      23.708333   ...
        ...           ...           ...         ...        ...         ...
        or
        Env              Interval   GDDs_Avg    ...
        DEH1_2020        1-9        16.693333   ... 
        DEH1_2020        10-13      18.155000   ...
        DEH1_2020        14-16      22.583333   ...
        DEH1_2020        17-19      19.321667   ...
        DEH1_2020        20-22      23.708333   ...
        ...               ...       ...         ...

    start_index : int
        This parameter is used to control the index position of the two different env dataframes

    """
    def __init__(self, phen, env, start_index=4,save_path='./Result/WinCorr/'):
        self.env = env
        self.phen = phen
        self.start_index = start_index
        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

    def sliding_window_corr(self, k, n, growth_period):
        """
        This is a sliding window calculation that is a fixed window size
        
        Parameters
        ----------
        k : int
            k is the size of the window

        n ： int
            n is the size of the crop growth cycle
            
        growth_period ： (int-int)
            The growth_period parameter is intended to divide the growth period into smaller
            intervals so that we can calculate localized growth throughout the growth period.

        Returns
        ----------
        results : {}
            Correlation results of all phenotypes with all environmental factors
        """
        results = {}

        env_indexs = self.env.columns[self.start_index:]
        lower_bound, upper_bound = growth_period
            
        for env_index in env_indexs:
            data = self.env[['Env', 'Leaf_Collar', env_index]]
            average = pd.DataFrame(WindCal.sliding_window(data, k, n)).T
            average.columns = average.iloc[0]
            average = average[1:].reset_index()
            average.columns = ['Env'] + list(average.columns[1:])

            phen_avgs = self.phen.columns[1:]
            env_avgs = average.columns[1:]

            correlation_data = []
            for phen_avg in phen_avgs:
                for env_avg in env_avgs:
                    valid_data = pd.concat([self.phen[phen_avg], average[env_avg]], axis=1).dropna()
                    if not valid_data.empty:
                        correlation_coefficient, _ = pearsonr(valid_data[phen_avg], valid_data[env_avg])
                        correlation_data.append({
                            'Phen_avg': phen_avg,
                            'Env_avg': env_avg,
                            'Correlation': correlation_coefficient
                        })

            # Convert the list of dictionaries to a DataFrame
            correlation_df = pd.DataFrame(correlation_data)
            split_values = correlation_df['Env_avg'].str.split('-', expand=True)
            split_values[0] = pd.to_numeric(split_values[0], errors='coerce')
            split_values[1] = pd.to_numeric(split_values[1], errors='coerce')
            correlation_df = correlation_df[(split_values[0] >= lower_bound) & (split_values[0] <= upper_bound) & (split_values[1] >= lower_bound) & (split_values[1] <= upper_bound)]
            results[env_index] = correlation_df

            # Save the DataFrame to a CSV file
            save_file_path = os.path.join(self.save_path, env_index + '.csv')
            correlation_df.to_csv(save_file_path, index=False)
                
        return results          

    def dynamic_window_corr(self):
        """
        This is a dynamic window calculation with an indeterminate window size

        Returns
        ----------
        results : {}
            Correlation results of all phenotypes with all environmental factors
        """
        
        results = {}
        env_indexs = self.env.columns[self.start_index:]
        for env_index in env_indexs:
            data = self.env.pivot(index='Env', columns='Leaf_Collar', values=env_index).reset_index().rename_axis(None, axis=1)
            phen_avgs = self.phen.columns[1:]
            env_avgs = data.columns[1:]

            correlation_data = []
            for phen_avg in phen_avgs:
                for env_avg in env_avgs:
                    valid_data = pd.concat([self.phen[phen_avg], data[env_avg]], axis=1).dropna()
                    if not valid_data.empty:
                        correlation_coefficient, _ = pearsonr(valid_data[phen_avg], valid_data[env_avg])
                        correlation_data.append({
                            'Phen_avg': phen_avg,
                            'Env_avg': env_avg,
                            'Correlation': correlation_coefficient
                        })

            # Convert the list of dictionaries to a DataFrame
            correlation_df = pd.DataFrame(correlation_data)

            # Save the DataFrame to a CSV file
            save_file_path = os.path.join(self.save_path, env_index + 'dynamic.csv')
            correlation_df.to_csv(save_file_path, index=False)
            results[env_index] = correlation_df
                
        return results