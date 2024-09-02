import pandas as pd
import glob
import re

def highlight_min_max(s):
    '''
    highlight the maximum in a Series red and minimum in green.
    '''
    if s[0] == '-' : return ['color: black' for _ in s]
    is_max = s == s.max()
    is_min = s == s.min()
    return ['color: red' if v else 'color: green' if c else '' for v, c in zip(is_max, is_min)]

def combine_stats(df, methods, augmentations):
    # Using a shorter approach to calculate mean and median for each configuration
    mean_median_stats = df.drop(columns = ['video']).agg(['mean', 'median']).T
    stats = {}
    for method in methods:
        # Calculate stats for each method
        method_stats = mean_median_stats.loc[[f"{method}{augmentation}" for augmentation in augmentations]].reset_index(drop=True).applymap("{:.1f}".format)
        method_stats['augmentation'] = augmentations
        stats[method] = method_stats[['mean', 'median']]

    # Combining the stats
    # Convert method_stats['augmentation'] to a DataFrame
    augmentation_df = pd.DataFrame(method_stats['augmentation'])

    # Concatenate stats DataFrames from the list comprehension
    stats_df = pd.concat([stats[method] for method in methods], axis=1)

    # Concatenate augmentation_df with stats_df side by side
    comb_stats = pd.concat([augmentation_df, stats_df], axis=1)
    comb_stats.columns = ["augmentations"]+[f'{method}_{stat}' for method in methods for stat in ['mean', 'median']]

    return comb_stats

# Read each file into a DataFrame
IID_pretrained = False

augmentations = ['', '_pseudo_dsms','_pseudo_dsms_automasking','_pseudo_dsms_automasking_noadjust', '_pseudo_dsms_automasking_sploss']  #['', '_pseudo', '_lightinput', '_pseudo_lightinput'] #['', '_add', '_rem', '_addrem'] #['']
methods = ['IID'] #['monodepth2', 'monovit', 'IID'] #

method_strings = "[!s]*" # *mono* or *IID*
clipped = True
distorted = False
singlescale = True
specscale = False
addedspec = True

if singlescale: scaling = "/singlescale" 
elif specscale: scaling = "/specular_scaling"
else:           scaling = ""

if addedspec:   addsptxt = "/addedspec"

dist = "dist_" if distorted else ""
dist_pre = "" if distorted else "un"

clip = "notclipped" if not clipped else ""

if IID_pretrained:
    method_ext = ["", "(.*?)results.*.csv"]
else:
    method_ext = ["/*/models", "finetuned_mono_hkfull_288(.*?)/models/"]    

direc = f"{dist_pre}disttrain/{dist_pre}dist{addsptxt}{scaling}" #"undisttrain/undist" or "disttrain/dist"
file_paths = sorted(glob.glob(f"/raid/rema/outputs/{direc}/{method_strings}{method_ext[0]}/*{clip}*.csv"))
pattern = fr"/outputs/{direc}/(.*?)/{dist}{method_ext[1]}"
        
dfs_mean_rmse = []
dfs_mean_rmse_masked = []

for file_path in file_paths:
    df = pd.read_csv(file_path)
    # Add a column with the method/file name
    method, model = re.findall(pattern, file_path)[0] # Extract method name from file path

    # Extract mean_rmse and mean_rmse_masked columns
    df_mean_rmse = df[['video', 'mean_rmse']]
    df_mean_rmse_masked = df[['video', 'mean_rmse_masked']]
    # Rename columns
    df_mean_rmse.columns = ['video', f"{method}{model}"]
    df_mean_rmse_masked.columns = ['video', f"{method}{model}"]
    dfs_mean_rmse.append(df_mean_rmse)
    dfs_mean_rmse_masked.append(df_mean_rmse_masked)

# Initialize the merged dataframes with the first dataframe in the list
merged_df_mean_rmse = dfs_mean_rmse[0]
merged_df_mean_rmse_masked = dfs_mean_rmse_masked[0]

# Merge the rest of the dataframes
for df in dfs_mean_rmse[1:]:
    merged_df_mean_rmse = merged_df_mean_rmse.merge(df, on='video')

for df in dfs_mean_rmse_masked[1:]:
    merged_df_mean_rmse_masked = merged_df_mean_rmse_masked.merge(df, on='video')

# Define the new order of the columns
new_order = ['video']
for method in methods:
    new_order.extend([f"{method}{augmentation}" for augmentation in augmentations])

# Reorder the columns
merged_df_mean_rmse = merged_df_mean_rmse[new_order]
merged_df_mean_rmse_masked = merged_df_mean_rmse_masked[new_order]

merged_df_mean_rmse.iloc[:,1:] = merged_df_mean_rmse.iloc[:,1:]/655.35
merged_df_mean_rmse_masked.iloc[:,1:] = merged_df_mean_rmse_masked.iloc[:,1:]/655.35

# Display the merged DataFrames
print("Mean RMSE:")
print(merged_df_mean_rmse)
print("\nMean RMSE Masked:")
print(merged_df_mean_rmse_masked)

stats = combine_stats(merged_df_mean_rmse, methods, augmentations).style.apply(highlight_min_max)
stats_masked= combine_stats(merged_df_mean_rmse_masked, methods, augmentations).style.apply(highlight_min_max)


# save to csv
output = f"/raid/rema/outputs/{direc}"
merged_df_mean_rmse.to_excel(f'{output}/{clip}{augmentations}results_rmse.xlsx', index=False)
merged_df_mean_rmse_masked.to_excel(f'{output}/{clip}{augmentations}results_rmse_masked.xlsx', index=False)
stats.to_excel(f'{output}/{clip}{augmentations}results_rmse_stats.xlsx', index=False)
stats_masked.to_excel(f'{output}/{clip}{augmentations}results_rmse_stats_masked.xlsx', index=False)
