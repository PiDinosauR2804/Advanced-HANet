import os
import json
from loguru import logger
import copy

input_path = 'Advanced-HANet\\output\\des'
output_path = 'Advanced-HANet\\output\\data_augment'
dataset = 'MAVEN'
NUM_PERM = 1

def augment_data(line):
    augment_data_list = []
    for key, value in line.items():
        for data in value:
            if 'events' in data:
                events = data['events']
                for event in events:
                    if 'description' in event:
                        description = event['description']
                        # Tạo dữ liệu mới bằng cách kết hợp text và description
                        new_data = copy.deepcopy(data)
                        new_data['text'] = new_data['text'] + ' ' + description
                        augment_data_list.append(new_data)
    return augment_data_list

def augment_dataset(input_path, output_path, dataset):
    os.path.exists(os.path.join(input_path))
    os.makedirs(output_path, exist_ok=True)
    for i in range(NUM_PERM):
        input_folder = os.path.join(input_path, dataset, 'perm'+str(i))
        output_folder = os.path.join(output_path, dataset, 'perm'+str(i))
        os.makedirs(output_folder, exist_ok=True)
        for file_name in os.listdir(input_folder):
            if not file_name.endswith('.jsonl'):
                continue
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)
            print(f"[START] Processing {file_name} in {output_file}")
            new_data = []
            with open(input_file, 'r') as f:
                for line in f:
                    augment_data = {}
                    #Đổi line thành dict
                    line = json.loads(line)
                    print(type(line))
                    aug_data = augment_data(line)
                    print(aug_data)
            
              

            