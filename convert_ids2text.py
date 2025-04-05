import json
import pandas as pd
from tqdm.notebook import tqdm
import os
from transformers import BertTokenizerFast

# Khởi tạo tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Đường dẫn đến các folder chứa dữ liệu
input_path = "data_incremental"
output_path = "raw_text"
datasets = ["ACE", "MAVEN"]

def ids2list(list_data:list)->list:
    res = []
    for item in list_data:
        # Chuyển đổi các ID thành văn bản
        text = tokenizer.decode(item['piece_ids'], skip_special_tokens=True)
        offsets = []
        for sp in item['span']:
            event = tokenizer.decode(item['piece_ids'][sp[0]: sp[1]+1], skip_special_tokens=True)
            offset = text.find(event)
            offsets.append([offset, offset + len(event)])
        
        res.append({"text": text, "offsets": offsets, "label": item['label']})
    return res
    

def convert(input_path:str, output_path:str, datasets:list)->None:
    os.makedirs(output_path, exist_ok=True)
    # Duyệt qua từng dataset
    for dataset in datasets:
        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(os.path.join(output_path, dataset), exist_ok=True)
        
        # Convert for train data
        for i in range(5):
            input_folder = os.path.join(input_path, dataset, "perm"+str(i))
            # Kiểm tra xem thư mục có tồn tại không
            if not os.path.exists(input_folder):
                print(f"Folder {input_folder} does not exist!")
                continue
            
            # Tạo thư mục đầu ra cho mỗi perm
            output_folder = os.path.join(output_path, dataset, "perm"+str(i))
            os.makedirs(output_folder, exist_ok=True)
            
            for file_name in os.listdir(input_folder):
                if file_name.endswith(".jsonl"):
                    input_file = os.path.join(input_folder, file_name)
                    output_file = os.path.join(output_path, dataset, "perm"+str(i), file_name)
                    new_data = None
                    
                    with open(input_file, 'r') as f:
                        data = [json.loads(line) for line in f]
                        new_data = ids2list(data)
                    
                    if new_data:
                        with open(output_file, 'w') as f:
                            for item in new_data:
                                f.write(json.dumps(item) + '\n')
                        print(f"Converted {input_file} to {output_file}")
                    else:
                        print(f"Error: No data found in {input_file}")
                    
        # Convert for test data
        input_file = os.path.join(input_path, dataset, f"{dataset}.test.jsonl")
        output_file = os.path.join(output_path, dataset, f"{dataset}.test.jsonl")
        new_data = None
        
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f]
            new_data = ids2list(data)
        
        if new_data:
            with open(output_file, 'w') as f:
                for item in new_data:
                    f.write(json.dumps(item) + '\n')
            print(f"Converted {input_file} to {output_file}")
        else:
            print(f"Error: No data found in {input_file}")
            
if __name__ == "__main__":
    convert(input_path, output_path, datasets)
    # convert(input_path, output_path, datasets)
        