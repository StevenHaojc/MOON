import pandas as pd
import numpy as np
import argparse
from os.path import join

def perform_ensemble(rootpath, model, fusion):
    num_classes = 3
    
    # Load results from all folds
    fold_dfs = [pd.read_csv(join(rootpath, model, fusion, f'fold{i}/test.csv')) for i in range(1, 6)]

    # Initialize fusion result dataframes
    fusion_dfs = {method: pd.DataFrame() for method in ['average', 'max', 'median']}

    # Initialize fusion dataframes with ID and True_Label from first fold
    if not fold_dfs[0].empty:
        for key in fusion_dfs.keys():
            fusion_dfs[key] = fold_dfs[0][['ID', 'True_Label']].copy()
            for class_index in range(num_classes):
                fusion_dfs[key][f'Score_Class_{class_index}'] = 0

    # Perform ensemble for each class
    for class_index in range(num_classes):
        score_column = f'Score_Class_{class_index}'
        class_scores = np.array([df[score_column] for df in fold_dfs])
        
        fusion_dfs['average'][score_column] = np.mean(class_scores, axis=0)
        fusion_dfs['max'][score_column] = np.max(class_scores, axis=0)
        fusion_dfs['median'][score_column] = np.median(class_scores, axis=0)

    # Save fusion results
    for method, df in fusion_dfs.items():
        df.to_csv(join(rootpath, model, fusion, f'{method}_fusion_test.csv'), index=False)

def main(args):
    for fusion in args.fusions:
        for model in args.model_types:
            perform_ensemble(args.rootpath, model, fusion) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process model and fusion method results.')
    parser.add_argument('--fusions', nargs='+', type=str, required=True, help='Fusion methods for ensemble')
    parser.add_argument('--rootpath', type=str, required=True, help='Root path for models and results')
    parser.add_argument('--model_types', nargs='+', type=str, required=True, help='Model types to process')
    
    args = parser.parse_args()
    main(args)