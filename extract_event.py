from llm import extract_event_gemini
import json
import pandas as pd
from tqdm.notebook import tqdm
import os
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Đường dẫn đến các folder chứa dữ liệu
input_path = "raw_text"
output_path = "data_incremental_by_llm"
datasets = ["MAVEN"]

def list2ids(list_data:list)->list:
    ids_data = []
    for item in list_data:
        event_list = extract_event_gemini(item['text'], model="gemini-2.0-flash", candidate=1)
        item['events'] = event_list
        # Chuyển đổi các văn bản thành ID
        opt = tokenizer(item['text'], return_offsets_mapping=True, truncation=True, max_length=512, padding="max_length")
        # Lấy các ID của các token
        piece_ids = opt['input_ids']
        # Lấy các span của các token
        offsets = opt['offset_mapping']
        # Lấy các span của các event word dựa vào offset mapping
        span = []
        for event in event_list:
            trigger_word = event['trigger_word']
            # Tìm vị trí của trigger word trong offsets
            start = -1
            end = -1
            for i, (s, e) in enumerate(offsets):
                if s <= item['text'].find(trigger_word) < e:
                    start = i
                    break
            if start != -1:
    

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
                    added_data = []
                    ids_data = []
                    
                    with open(input_file, 'r') as f:
                        for line in f:
                            added_line = {}
                            ids_line = {}
                            # Chuyển đổi từng dòng JSON thành dict
                            json_line = json.loads(line)
                            for key, value in json_line.items():
                                added_line[key], ids_line[key] = list2ids(value)
                            
                            added_data.append(added_line)
                            ids_data.append(ids_line)
                    
                    if added_data:
                        with open(input_file, 'w') as f:
                            for item in added_data:
                                f.write(json.dumps(item) + '\n')
                        print(f"Added data converted and saved to {input_file}")
                    else:
                        print(f"Error: No data found in {input_file}")
                    
                    if ids_data:
                        with open(output_file, 'w') as f:
                            for item in ids_data:
                                f.write(json.dumps(item) + '\n')
                        print(f"IDs data converted and saved to {output_file}")
                    else:
                        print(f"Error: No data found in {input_file}")
                    
        # Convert for test data
        input_file = os.path.join(input_path, dataset, f"{dataset}.test.jsonl")
        output_file = os.path.join(output_path, dataset, f"{dataset}.test.jsonl")
        added_data = None
        ids_data = None
        
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f]
            added_data, ids_data = list2ids(data)
        
        if added_data:
            with open(input_file, 'w') as f:
                for item in ids_data:
                    f.write(json.dumps(item) + '\n')
            print(f"Added data converted and saved to {input_file}")
        else:
            print(f"Error: No data found in {input_file}")
            
        if ids_data:
            with open(output_file, 'w') as f:
                for item in added_data:
                    f.write(json.dumps(item) + '\n')
            print(f"IDs data converted and saved to {output_file}")
        else:
            print(f"Error: No data found in {input_file}")
        
  
if __name__ == "__main__":
    convert(input_path, output_path, datasets)
    # convert(input_path, output_path, datasets)
        