# Convert filenames to include labels
input_txt_file = '/home/lichunli.lcl/LCL/ESCode/New/main/txtfile/train_val.txt'
output_txt_file = '/home/lichunli.lcl/LCL/ESCode/New/main/txtfile/train_val_lab.txt'

def get_label(prefix):
    labels = {
        'V1': '0',
        'V2': '1', 
        'V3': '2'
    }
    return labels.get(prefix)

with open(input_txt_file, 'r') as infile, open(output_txt_file, 'w') as outfile:
    for line in infile:
        filename = line.strip()
        prefix = filename.split('_')[0]
        label = get_label(prefix)
        if label is not None:
            outfile.write(f"{filename}\t{label}\n")
        else:
            print(f"Warning: Invalid prefix '{prefix}' in filename '{filename}'")

print(f"File names and labels have been written to {output_txt_file}")