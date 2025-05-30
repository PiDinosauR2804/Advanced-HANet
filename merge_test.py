import os
import json
import numpy as np

dataset = 'ACE'

original_path = "./data/data_ids_enhence"
span_path = "./output/des"

with open(os.path.join(original_path, dataset, f"{dataset}.test.jsonl"), 'r') as f:
    original_test = [json.loads(line) for line in f]
    
with open(os.path.join(span_path, dataset, f"{dataset}.test.jsonl"), 'r') as f:
    span_test = [json.loads(line) for line in f]

# for oitem, sitem in zip(original_test, span_test):
#     label_mask = []
#     for sp in oitem['span']:
#         if sp in sitem['span']:
#             label_mask.append(True)
#         else:
#             label_mask.append(False)
            
#     oitem["label_mask"] = label_mask
    
# with open(os.path.join(original_path, dataset, f"{dataset}.test.jsonl"), 'w') as f:
#     for item in original_test:
#         f.write(json.dumps(item) + "\n")

o_len = 0
s_len = 0
for oitem, sitem in zip(original_test, span_test):
    o_len += len(oitem['span'])
    s_len += len(sitem['span'])
    
print(f"Original span length: {o_len}")
print(f"Span span length: {s_len}")