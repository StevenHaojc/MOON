import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

def copy_file(inpn, outpn):
    shutil.copy(inpn, outpn)

inp1 = '/mnt/workspace/workgroup/LCL/Test/ESC_NC_Screening/nnUNet_raw_data_base/nnUNet_raw_data/Task007_Esc/labelsTs'
outp1 = '/mnt/workspace/workgroup/LCL/Test/ESC_NC_Screening/Newdata/mask/esophagus'

# List of files to copy
files_to_copy = os.listdir(inp1)

# Get the number of CPU cores
cpu_cores = multiprocessing.cpu_count()

# Using a reasonable number of workers based on your system's capability and the I/O characteristics
max_workers = cpu_cores

# Create a ThreadPoolExecutor with a dynamic number of workers based on your system's CPU cores
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Create a list to hold the future results
    futures = []
    # Submit tasks to the executor
    for i in files_to_copy:
        inpn = os.path.join(inp1, i)
        outpn = os.path.join(outp1, i)
        futures.append(executor.submit(copy_file, inpn, outpn))

    # Create progress bar with the `tqdm` library
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Copying files", unit="file"):
        pass  # No operation needed; progress will update automatically
