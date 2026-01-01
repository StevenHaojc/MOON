import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import kendalltau

def convert_labels_for_one_vs_rest(true_labels, vs_label):
    return [0 if label < vs_label else 1 for label in true_labels]

def evaluate_results(df, vs_labels):
    results = {}
    auc_scores = []
    for vs_label in vs_labels:
        converted_labels = convert_labels_for_one_vs_rest(df['True_Label'], vs_label)
        accuracy = accuracy_score(converted_labels, df['Predicted_Label'] >= vs_label)
        auc_score = roc_auc_score(converted_labels, df[f'Score_Class_{vs_label}'])
        auc_scores.append(auc_score)
        results[f'acc_{vs_label}_vs_rest'] = accuracy
        results[f'auc_{vs_label}_vs_rest'] = auc_score*100
    all_acc = accuracy_score(df['True_Label'], df['Predicted_Label'])
    k_accuracy, _ = kendalltau(df['True_Label'], df['Predicted_Label'])
    results['all_auc'] = sum(auc_scores) / len(auc_scores)
    results['all_acc'] = all_acc*100
    results['kenda_acc'] = k_accuracy*100
    return results

def main(args):
    rootpath = args.rootpath
    all_results = pd.DataFrame()

    for h in 'valid', 'test':
        csvpath = f"{rootpath}/{h}.csv"
        df = pd.read_csv(csvpath)
        results = evaluate_results(df, vs_labels=[1, 2, 3])
        results['dataset'] = h
        all_results = pd.concat([all_results, pd.DataFrame([results])], ignore_index=True)
    all_results.to_csv(f"{rootpath}/{'summary_validtest_mtrc3.csv'}", index=False)

parser = argparse.ArgumentParser(description='Process model and fusion method results.')
parser.add_argument('--rootpath', type=str, required=True, help='Root path where the models and results are stored')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)