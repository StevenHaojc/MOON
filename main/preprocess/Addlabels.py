input_txt_file = '/mnt/workspace/workgroup/LCL/Code/ESC_newCode/txtfile/1/train_val.txt'

output_txt_file = '/mnt/workspace/workgroup/LCL/Code/ESC_newCode/txtfile/1/train_val_lab.txt'

def get_label(prefix):
    return '0' if prefix in ['V0', 'V1'] else '1' if prefix in ['V2', 'V3'] else None

with open(input_txt_file, 'r') as infile, open(output_txt_file, 'w') as outfile:
    for line in infile:
        filename = line.strip()  
        prefix = filename.split('_')[0]  
        label = get_label(prefix)
        outfile.write(f"{filename}\t{label}\n") 

print(f"File names and labels have been written to {output_txt_file}")
