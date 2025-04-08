from llm import extract_event_gemini
import json
import pandas as pd
from tqdm.notebook import tqdm
import os
from transformers import BertTokenizerFast
import time

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Đường dẫn đến các folder chứa dữ liệu
input_path = "raw_text"
output_path = "data_incremental_by_llm"
datasets = ["MAVEN"]
NUM_TRY = 3

def list2ids(list_data:list)->list:
    ids_data = []
    for item in list_data:
        for i in range(NUM_TRY):
            try:
                event_list = extract_event_gemini(item['text'], model="gemini-2.0-flash", candidate=1)[0]
                break
            except Exception as e:
                print(f"Attempt {i}/{NUM_TRY} failed for text:\n{item['text']}\nError: {e}")
                time.sleep(15)  # Thời gian chờ giữa các lần thử
        else:
            print(f"All attempts failed for text:\n{item['text']}")
            continue
        
        # Nếu không tìm thấy event word, bỏ qua item này
        if not event_list:
            continue

        # Chuyển đổi các văn bản thành ID
        opt = tokenizer(item['text'], return_offsets_mapping=True, truncation=True, max_length=512, padding="max_length")
        # Lấy các ID của các token
        piece_ids = opt['input_ids']
        # Lấy các span của các token
        offsets_mp = opt['offset_mapping']
        # Lấy các span của các event word dựa vào offset mapping
        span = []
        for event in event_list:
            trigger_word = event['trigger_word']
            # Tìm vị trí của trigger word trong offsets_mp
            start = -1
            end = -1
            for i, (s, e) in enumerate(offsets_mp):
                if s <= item['text'].find(trigger_word) < e:
                    start = i
                    break
            if start != -1:
                for i, (s, e) in enumerate(offsets_mp[start:], start):
                    if s >= item['text'].find(trigger_word) + len(trigger_word):
                        end = i
                        break
            if start != -1 and end != -1:
                span.append((start, end))
 
        ids_data.append({
            'text': item['text'],
            'events': event_list,
            'span': span,
            'piece_ids': piece_ids,
            'offsets': item['offsets']
        })
        
    return ids_data
    

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
                    ids_data = []
                    
                    with open(input_file, 'r') as f:
                        for line in f:
                            added_line = {}
                            ids_line = {}
                            # Chuyển đổi từng dòng JSON thành dict
                            json_line = json.loads(line)
                            for key, value in json_line.items():
                                ids_line[key] = list2ids(value)
                            
                            added_data.append(added_line)
                            ids_data.append(ids_line)
                    
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
        ids_data = None
        
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f]
            ids_data = list2ids(data)
        
            
        if ids_data:
            with open(output_file, 'w') as f:
                for item in ids_data:
                    # Chuyển đổi từng dòng JSON thành dict
                    f.write(json.dumps(item) + '\n')
            print(f"IDs data converted and saved to {output_file}")
        else:
            print(f"Error: No data found in {input_file}")
        
if __name__ == "__main__":
    convert(input_path, output_path, datasets)
    # convert(input_path, output_path, datasets)
        