from data_process.llm import Extractor, Extractor_Gemini, is_quota_exhausted_error, test_extractor
from data_process.hash_text import get_text_hash, load_status, save_status, is_duplicate_hashes
from data_process.convert import sent2ids
from api_key import GEMINI_KEY
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

INPUT_PATH = "raw_text"
OUT_PATH = "test2"
DATASETS = ["MAVEN"]
NUM_TRY = 4
MAX_CONSECUTIVE_429_ERROR = 6

def filter_valid_extractors():
    print("Testing API keys...")
    valid_extractors = []
    for idx, key in enumerate(GEMINI_KEY):
        extractor = Extractor(api_key=key)
        if test_extractor(extractor):
            print(f"[OK] API key {idx} is valid.")
            valid_extractors.append(extractor)
        else:
            print(f"[FAILED] API key {idx} is exhausted (429). Skipping...")
    return valid_extractors

# Một extractor Gemini tương ứng với một API Key
extractors = filter_valid_extractors()
NUM_WORKERS = len(extractors)

if NUM_WORKERS == 0:
    raise RuntimeError("No valid API keys available.")

def process_item(item, extractor_id=0):
    extractor = extractors[extractor_id]
    try:
        event_list = extractor.extract_event(item['text'], model="gemini-2.0-flash", candidate=1)[0]
        if event_list:
            new_item = {
                'text': item['text'],
                'events': event_list,
            }
            new_item = sent2ids([new_item])[0]
            return new_item

    except Exception as e:
        if is_quota_exhausted_error(e):
            raise RuntimeError(f"429 Error - Quota exhausted - extractor_id={extractor_id}")
        else:
            raise e  # Ném lại lỗi khác để xử lý tiếp

def sents2ids_parallel(list_data: list[dict], status_file: str, resume: bool = True):
    if is_duplicate_hashes(list_data):
        print("Duplicate hashes found in the input data.")
        return list_data
    
    ids_data = []
    status_done = load_status(status_file)
    quota_exceeded_count = [0] * NUM_WORKERS  # Chỉ đếm lỗi 429 liên tiếp

    def task(item, idx):
        text_hash = get_text_hash(item['text'])
        if resume and text_hash in status_done:
            return None
        
        for attempt in range(NUM_TRY):
            for i, extractor in enumerate(extractors):
                try:
                    result = process_item(item, extractor_id=i)
                    if result and result['events']:
                        save_status(status_file, item['text'])
                        quota_exceeded_count[i] = 0  # Reset nếu thành công
                        return result
                    else:
                        quota_exceeded_count[i] = 0  # Reset nếu không lỗi 429

                except Exception as e:
                    # RuntimeError ở đây đại diện cho lỗi 429 RESOURCE_EXHAUSTED
                    if is_quota_exhausted_error(e):
                        quota_exceeded_count[i] += 1
                        if quota_exceeded_count[i] >= MAX_CONSECUTIVE_429_ERROR:
                            print(f"[FATAL] Quota exhausted for API key {i} after {MAX_CONSECUTIVE_429_ERROR:} consecutive 429 errors.")
                            raise SystemExit("Stopping due to quota exhaustion.")
                    else:
                        quota_exceeded_count[i] = 0
                    if attempt < NUM_TRY - 1:
                        print(f"Error (attempt {attempt}) on item: {item['text'][:30]}... | Error: {e}")
                        time.sleep(15)
                    else:
                        print(f"Max attempts reached for item: {item['text'][:30]}... | Error: {e}")
                        break    

        return None


    ids_data = [None] * len(list_data)  # Khởi tạo list kết quả với độ dài tương ứng
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_idx = {executor.submit(task, item, idx): idx for idx, item in enumerate(list_data)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            if result:
                ids_data[idx] = result  # Gán vào đúng vị trí theo index
                
    for item1, item2 in zip(list_data, ids_data):
        if item2['text'] != item1['text']:
            print(f"Error: Text mismatch after processing. Original: {item1['text'][:30]}... | Processed: {item2['text'][:30]}...")
                
            return list_data  # Trả về dữ liệu gốc nếu có sự khác biệt trong văn bản
            
    return ids_data

def convert(input_path: str, output_path: str, datasets: List[str], resume=True):
    os.makedirs(output_path, exist_ok=True)
    for dataset in datasets:
        os.makedirs(os.path.join(output_path, dataset), exist_ok=True)

        for i in range(5):
            input_folder = os.path.join(input_path, dataset, "perm"+str(i))
            if not os.path.exists(input_folder):
                continue

            output_folder = os.path.join(output_path, dataset, "perm"+str(i))
            os.makedirs(output_folder, exist_ok=True)

            for file_name in os.listdir(input_folder):
                if not file_name.endswith(".jsonl"):
                    continue

                input_file = os.path.join(input_folder, file_name)
                output_file = os.path.join(output_folder, file_name)
                status_file = output_file + ".status.jsonl"

                ids_data = []
                with open(input_file, 'r') as f:
                    for line in f:
                        json_line = json.loads(line)
                        for key, value in json_line.items():
                            if not isinstance(value, list):
                                print(f"Skipping key {key} - not a list")
                                continue
                            ids_line = list2ids_parallel(value, status_file, resume=resume)
                            ids_data.append({key: ids_line})


                if ids_data:
                    with open(output_file, 'w') as f:
                        for item in ids_data:
                            if item is not None:                        
                                f.write(json.dumps(item) + '\n')

        # Convert test
        input_file = os.path.join(input_path, dataset, f"{dataset}.test.jsonl")
        output_file = os.path.join(output_path, dataset, f"{dataset}.test.jsonl")
        status_file = output_file + ".status.jsonl"

        if os.path.exists(input_file):
            with open(input_file, 'r') as f:
                data = [json.loads(line) for line in f]
                ids_data = list2ids_parallel(data, status_file, resume=resume)
                if ids_data:
                    with open(output_file, 'w') as f:
                        for item in ids_data:
                            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    convert(INPUT_PATH, OUT_PATH, DATASETS, resume=True)
