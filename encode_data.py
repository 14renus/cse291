import pandas as pd
import os
from dataloading import *

### write encoded forms to files for easier reads/processing
data_dir = 'data/numerical_data_set_simple'
output_dir = "data/numerical_data_set_simple_torch"

data_dir = 'data/numerical_data_set_simple'

filenames = []
filenames_by_type = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[]}
for file in os.listdir(data_dir):
    filename, file_extension = os.path.splitext(file)
    if file_extension==".csv" and '._labelled' not in filename:
        filenames.append(file)
        typ = filename[-1]
        if typ=='D' and 'gen' in filename:
            continue
        filenames_by_type[typ].append(file)
print(filenames)
print()
print(filenames_by_type)


#filenames=['labelled_gen_data15_E.csv']
#for file in filenames_by_type['D']:
#for file in ['labelled_extr_data1_A.csv']:
for typ in filenames_by_type:
    for file in filenames_by_type[typ]:
        print(file)
        filename, file_extension = os.path.splitext(file)
        df = pd.read_csv(os.path.join(data_dir,file))

        values = df['Attribute_value']
        targets = df['Numerical_value']

        lim = 200//len(filenames_by_type[typ])

        inputs = prepare_data(values[lim:],padding_len=22)
        outputs = prepare_targets(targets[lim:],padding_len=22)

        torch.save([inputs,outputs],os.path.join(output_dir,filename))
        q = torch.load(os.path.join(output_dir,filename))

        #assertEqual(inputs,q[0])
        #assertEqual(outputs,q[1])
        #print(inputs==q[0])
        #print(outputs==q[1])

        print(filename)

print('done')
