import pandas as pd
import os
import argparse

def merge_csv_multi1(csv_file_path):
    df = pd.read_csv(csv_file_path)
    print(df)
    fusiontype = df['fusion'].unique()
    dataset = df['dataset'].unique()
    modeltype = df['type'].unique()
    mean_values_df = pd.DataFrame()
    for f in fusiontype:
        for d in dataset:
            for t in modeltype:
                filtered_df = df[(df['dataset'] == d) & (df['fusion'] == f) & (df['type'] == t)]
                mean_values = filtered_df.mean(numeric_only=True)
                mean_values_df_temp = pd.DataFrame(mean_values).transpose()
                mean_values_df_temp['dataset'] = d
                mean_values_df_temp['fusion'] = f
                mean_values_df_temp['type'] = t
                mean_values_df = pd.concat([mean_values_df, mean_values_df_temp], ignore_index=True)
    return mean_values_df
def merge_csv_single1(csv_file_path):
    df = pd.read_csv(csv_file_path)
    dataset = df['dataset'].unique()
    modeltype = df['type'].unique()
    mean_values_df = pd.DataFrame()
    for d in dataset:
        for t in modeltype:
            filtered_df = df[(df['dataset'] == d) & (df['type'] == t)]
            mean_values = filtered_df.mean(numeric_only=True)
            mean_values_df_temp = pd.DataFrame(mean_values).transpose()
            mean_values_df_temp['dataset'] = d
            mean_values_df_temp['type'] = t
            mean_values_df_temp['fusion'] = 'single'
            mean_values_df = pd.concat([mean_values_df, mean_values_df_temp], ignore_index=True)
    return mean_values_df


def merge_csv_multi2(csv_file_path):
    df = pd.read_csv(csv_file_path)
    fusiontype = df['fusion_type'].unique()
    dataset = df['fusion_method'].unique()
    modeltype = df['model'].unique()
    mean_values_df = pd.DataFrame()
    for f in fusiontype:
        for d in dataset:
            for t in modeltype:
                filtered_df = df[(df['fusion_method'] == d) & (df['fusion_type'] == f) & (df['model'] == t)]
                mean_values = filtered_df.mean(numeric_only=True)
                mean_values_df_temp = pd.DataFrame(mean_values).transpose()
                mean_values_df_temp['dataset'] = d
                mean_values_df_temp['fusion'] = f
                mean_values_df_temp['type'] = t
                mean_values_df = pd.concat([mean_values_df, mean_values_df_temp], ignore_index=True)
    return mean_values_df




def merge_csv_single2(csv_file_path):
    df = pd.read_csv(csv_file_path)
    fusion_method = df['fusion_method'].unique()
    modeltype = df['organ_type'].unique()
    mean_values_df = pd.DataFrame()
    for d in fusion_method:
        for t in modeltype:
            filtered_df = df[(df['fusion_method'] == d) & (df['organ_type'] == t)]
            mean_values = filtered_df.mean(numeric_only=True)
            mean_values_df_temp = pd.DataFrame(mean_values).transpose()
            mean_values_df_temp['dataset'] = 'test'
            mean_values_df_temp['type'] = t
            mean_values_df_temp['fusion'] = d
            mean_values_df = pd.concat([mean_values_df, mean_values_df_temp], ignore_index=True)
    return mean_values_df





def main(args):
    single_csvp1 = os.path.join(args.csv_file_path, 'SingleOrgan/summary_validtest.csv')
    multi_csvp1 = os.path.join(args.csv_file_path, 'ThreeOrgan/summary_validtest.csv')
    single_csvp2 = os.path.join(args.csv_file_path, 'SingleOrgan/ensemble_results.csv')
    multi_csvp2 = os.path.join(args.csv_file_path, 'ThreeOrgan/ensemble_results.csv')
    mean_values_dfs1 = merge_csv_single1(single_csvp1)
    mean_values_dfm1 = merge_csv_multi1(multi_csvp1)
    mean_values_dfs2 = merge_csv_single2(single_csvp2)
    mean_values_dfm2 = merge_csv_multi2(multi_csvp2)    
    all_mean_values_df = pd.concat([mean_values_dfs1, mean_values_dfm1, mean_values_dfs2, mean_values_dfm2], ignore_index=True)
    
    output_csv_path = os.path.join(args.csv_file_path, 'merged_summary.csv')
    all_mean_values_df.to_csv(output_csv_path, index=False)

    return all_mean_values_df

parser = argparse.ArgumentParser(description='Merge CSV results and calculate mean values.')
parser.add_argument('--csv_file_path', type=str, required=True, help='Root path where the CSV files are stored')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)



