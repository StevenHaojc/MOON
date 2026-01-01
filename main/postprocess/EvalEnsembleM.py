import pandas as pd
import numpy as np
import argparse
from evaluate_Multi_results import evaluate_results

def prediction2label(pred):
    """Convert prediction scores to class labels using ordinal classification"""
    pred_tran = (pred > 0.5).cumprod(axis=1)
    labels = pred_tran.sum(axis=1) - 1
    return labels

def main(args):
    # Setup paths and parameters
    rootpath = args.rootpath
    layer_fusions = args.fusion
    model_types = args.model_types
    fusion_methods = ['average', 'max', 'median']
    
    # Initialize results dataframe
    all_results = pd.DataFrame()

    # Process each model configuration
    for model_type in model_types:
        for layer_fusion in layer_fusions:
            for method in fusion_methods:
                # Read predictions
                csv_path = f"{rootpath}/{model_type}/{layer_fusion}/{method}_fusion_test.csv"
                df = pd.read_csv(csv_path)
                
                # Get predictions
                pred_scores = df[['Score_Class_0', 'Score_Class_1', 'Score_Class_2']].values
                df['Predicted_Label'] = prediction2label(pred_scores)
                
                # Evaluate results
                results = evaluate_results(df, vs_labels=[1, 2])
                
                # Add metadata
                results['model'] = model_type
                results['fusion_method'] = method 
                results['fusion_type'] = layer_fusion
                
                # Append to overall results
                all_results = pd.concat([all_results, pd.DataFrame([results])], 
                                      ignore_index=True)

    # Save final results
    all_results.to_csv(f"{rootpath}/ensemble_results.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process model and fusion method results.')
    parser.add_argument('--fusion', nargs='+', type=str, required=True, 
                        help='Layer fusion methods')
    parser.add_argument('--rootpath', type=str, required=True,
                        help='Root path for data')
    parser.add_argument('--model_types', nargs='+', type=str, required=True,
                        help='Model types to evaluate')
    
    args = parser.parse_args()
    main(args)