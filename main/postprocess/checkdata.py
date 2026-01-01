import pandas as pd

def read_csvfile(inpath):
    # Read CSV file and return max 'acc' value
    df = pd.read_csv(inpath) 
    return df['acc'].max()

# Store accuracy for each fold
results = {}
fusionmethod = 'concat'
for fid in ('fold1', 'fold2', 'fold3'):
    base_path = f'/home/lichunli.lcl/LCL/ESCode/New/results/MultiOrgan3cls/Inter/{fusionmethod}/{fid}/valid_metrics.csv'
    cross_path = f'/home/lichunli.lcl/LCL/ESCode/New/results/MultiOrgan3cls/Cross0/{fusionmethod}/{fid}/valid_metrics.csv'
    ncross_path = f'/home/lichunli.lcl/LCL/ESCode/New/results/MultiOrganOgm/Weight/{fusionmethod}/OGM_GE_0.1_0_50/{fid}/valid_metrics.csv'

    # Get accuracies 
    base_acc = read_csvfile(base_path)
    cross_acc = read_csvfile(cross_path)
    ncross_acc = read_csvfile(ncross_path)
    
    # Calculate difference
    avg_acc = ncross_acc - base_acc
    
    results[fid] = {
        'base': base_acc,
        'cross': cross_acc, 
        'ncross': ncross_acc,
        'cross_base': avg_acc
    }

# Print results
for fid, accs in results.items():
    print(f"Fold: {fid}, Base accuracy: {accs['base']}, Cross accuracy: {accs['cross']}, "\
          f"nCross accuracy: {accs['ncross']}, cross_base accuracy: {accs['cross_base']}")