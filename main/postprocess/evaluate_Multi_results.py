import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

def convert_labels_for_one_vs_rest(true_labels, vs_label):
    return [0 if label < vs_label else 1 for label in true_labels]

def evaluate_results(df, vs_labels):
    results = {}
    
    # Calculate metrics for each one-vs-rest case
    for vs_label in vs_labels:
        converted_labels = convert_labels_for_one_vs_rest(df['True_Label'], vs_label)
        
        # Calculate accuracy and AUC
        accuracy = accuracy_score(converted_labels, df['Predicted_Label'] >= vs_label)
        auc_score = roc_auc_score(converted_labels, df[f'Score_Class_{vs_label}'])
        
        results[f'acc_{vs_label}_vs_rest'] = accuracy
        results[f'auc_{vs_label}_vs_rest'] = auc_score

    # Calculate overall accuracy
    results['all_acc'] = accuracy_score(df['True_Label'], df['Predicted_Label'])
    
    return results

def main(args):
    rootpath = args.rootpath
    fusions = args.fusions
    
    # Initialize results DataFrame
    all_results = pd.DataFrame()
    
    # Process each combination of parameters
    for fusion in fusions:
        for fold in ('fold1', 'fold2', 'fold3', 'fold4', 'fold5'):
            for model in args.model_types:
                for dataset in ('valid', 'test'):
                    # Read and evaluate results
                    csvpath = f"{rootpath}/{model}/{fusion}/{fold}/{dataset}.csv"
                    df = pd.read_csv(csvpath)
                    
                    results = evaluate_results(df, vs_labels=[1, 2])
                    
                    # Add metadata
                    results.update({
                        'fold': fold,
                        'type': model,
                        'dataset': dataset,
                        'fusion': fusion
                    })
                    
                    # Append to results
                    all_results = pd.concat([all_results, pd.DataFrame([results])], 
                                          ignore_index=True)
    
    # Save summary results
    all_results.to_csv(f"{rootpath}/summary_validtest.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process model and fusion method results.')
    parser.add_argument('--fusions', nargs='+', type=str, required=True,
                        help='Fusion methods for evaluation')
    parser.add_argument('--rootpath', type=str, required=True,
                        help='Root path for data')
    parser.add_argument('--model_types', nargs='+', type=str, required=True,
                        help='Model types to evaluate')
    
    args = parser.parse_args()
    main(args)