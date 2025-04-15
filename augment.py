import os
import json
from loguru import logger
import copy

input_path = 'output\des'
output_path = 'output\augmented'
dataset = 'MAVEN'
NUM_PERM = 1

def augment_data(line: dict):
    new_line = {}
    for key, value in line:
        new_line[key] = []
        for data in value:
            new_data = copy.deepcopy(data)
            #Concat text và des để tạo thành 1 data mới 
            new_data['text'] = new_data['text'] + ' ' + new_data['description']
            new_line[key].append(new_data)
    return new_line

def augment_dataset(input_path, output_path, dataset):
    os.path.exists(input_path)
    os.makedirs(output_path, exist_ok=True)
    for i in range(NUM_PERM):
        input_folder = os.path.join(input_path, dataset, 'perm'+str(i))
        output_folder = os.path.join(output_path, dataset, 'perm'+str(i))
        os.makedirs(output_folder, exist_ok=True)
        for file_name in os.listdir(output_folder):
            if not file_name.endswith('.jsonl'):
                continue
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)
            new_data = []
            with open(input_file, 'r') as f:
                for line in f:
                    augment_data = {}
                    #Đổi line thành dict
                    line = json.loads(line)
                    #Gọi hàm augment_data để tạo ra dữ liệu mới
                    augment_data = augment_data(line)
                    #Ghi dữ liệu mới vào file
                    new_data.append(augment_data)
            
            with open(output_file, 'w') as f:
                for line in new_data:
                    f.write(json.dumps(line) + '\n')
            logger.info(f"Augmented {file_name} and saved to {output_file}")
            logger.info(f"[FINISHED] Processing {file_name} in {output_file}")

if __name__ == "__main__":
    # Check if input path exists
    if not os.path.exists(input_path):
        logger.error(f"[ERROR] Input path {input_path} does not exist", mode="ERROR")
    else:
        # Check if output path exists, if not create it
        os.makedirs(output_path, exist_ok=True)
        augment_dataset(input_path, output_path, dataset)                       