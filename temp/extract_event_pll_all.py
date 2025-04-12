from extractor.llm import Extractor, Extractor_Gemini, is_quota_exhausted_error
from extractor.extractor_config import parse_arguments
from utils.convert import sent2ids
from extractor.api_key import GEMINI_KEY
import json
import os
import time
from typing import List
import threading
import queue
import loguru

# Global variables
MAX_LEN = 30
INPUT_ROOT = "raw_text"
OUTPUT_ROOT = "test3"
DATASETS = ["MAVEN"]
NUM_TRY = 3
MAX_CONSECUTIVE_429_ERROR = 3
LOGGING_FOLDER = "logs/extractor"

# Tạo thư mục log nếu chưa có
os.makedirs(LOGGING_FOLDER, exist_ok=True)

# Cấu hình logging
loguru.logger.add(os.path.join(LOGGING_FOLDER, "extractor.log"), rotation="1 MB", retention="10 days", level="INFO")
loguru.logger.info("Start logging...")

def log(str:str, mode="INFO"):
    with print_lock:
        if mode == "INFO":
            loguru.logger.info(str)
        elif mode == "ERROR":
            loguru.logger.error(str)
        elif mode == "WARNING":
            loguru.logger.warning(str)
        elif mode == "DEBUG":
            loguru.logger.debug(str)
        elif mode == "FATAL":
            loguru.logger.critical(str)

# Global variables for threading
task_queue = queue.Queue()
results = []
stop_event = threading.Event()
print_lock = threading.Lock()
extractors = []

for i in range(len(GEMINI_KEY)):
    extractors.append({
        'extractor': Extractor_Gemini(api_key=GEMINI_KEY[i]),
        'consecutive_429_error': 0,
    })

# Function to check if the extractor is available
processed_item = 0
remained_item = 0
def worker(worker_id):
    global task_queue
    global results
    global stop_event
    global processed_item
    global remained_item
    extractor_id = worker_id % len(extractors)
    extractor = extractors[extractor_id]
    
    while not stop_event.is_set():
        if extractor['consecutive_429_error'] >= MAX_CONSECUTIVE_429_ERROR:
            log(f"[FATAL] Worker {worker_id} got error 429 more than {MAX_CONSECUTIVE_429_ERROR} times. Stoping ...", mode="FATAL")
            # Stop the thread
            break
        
        new_item = None
        item = None
        try:
            idx, item = task_queue.get(timeout=0.1)
            if 'events' in item:
                log(f"[SKIP] Worker {worker_id} processed item: {item['text'][:MAX_LEN]}")
                results.append((idx, item))
                continue
            
            # Try to process the item
            for i in range(NUM_TRY):
                try:
                    event_list = extractor['extractor'].extract_event(item['text'], model="gemini-2.0-flash", candidate=1)[0]
                    extractor['consecutive_429_error'] = 0
                    if event_list:
                        new_item = {
                            'text': item['text'],
                            'events': event_list,
                        }
                        new_item = sent2ids(new_item) # Add piece_ids, span and offsets
                        break
                    
                except Exception as e:
                    if is_quota_exhausted_error(e):
                        log(f"[429 ERROR at ATTEMPT {i+1}/{NUM_TRY}] Worker {worker_id} got 429 error", mode="WARNING")
                        extractor['consecutive_429_error'] += 1
                        time.sleep(20)  # Wait for 15 seconds before retrying
                    else:
                        extractor['consecutive_429_error'] = 0
                        log(f"[ERROR at ATTEMPT {i+1}/{NUM_TRY}] Worker {worker_id} got error: {e}", mode="ERROR")
                            
                
        except queue.Empty:
            time.sleep(1)
        except Exception as e:
            log(f"[FATAL] Worker {worker_id} got error: {e}", mode="FATAL")
            continue
        
        if new_item:
            # If we successfully processed the item, add it to the results
            log(f"[SUCCESS] Worker {worker_id} processed item: {new_item['text'][:MAX_LEN]}")
            results.append((idx, new_item))
            processed_item += 1
        elif item:
            # If we exit the for loop without breaking, it means we have exhausted all attempts
            log(f"[SKIP] Worker {worker_id} cannot process: {item['text'][:MAX_LEN]}", mode="WARNING")
            results.append((idx, item))
            remained_item += 1
            
# Init threading
threads = []
for i in range(len(GEMINI_KEY)):
    threads.append(threading.Thread(target=worker, args=(i,)))
    threads[i].start()
 

def convert(input_path: str, output_path: str, datasets: List[str], resume=False):
    global results
    global task_queue
    global threads
    global stop_event
    global processed_item
    global remained_item
    
    # Resume from output_path if resume is True
    if resume:
        input_path = output_path
    else:
        os.makedirs(output_path, exist_ok=True)
    
    # Check if at least one thread is running
    def _is_any_thread_runing():
        return any(thread.is_alive() for thread in threads)
    
    # Convert dataset
    for dataset in datasets:
        os.makedirs(os.path.join(output_path, dataset), exist_ok=True)

        for i in range(1, 2):
            input_folder = os.path.join(input_path, dataset, "perm"+str(i))
            if not os.path.exists(input_folder):
                log(f"[ERROR] Folder {input_folder} is not exist", mode="ERROR")
                continue

            output_folder = os.path.join(output_path, dataset, "perm"+str(i))
            os.makedirs(output_folder, exist_ok=True)

            for file_name in os.listdir(input_folder):
                if not file_name.endswith(".jsonl"):
                    continue

                input_file = os.path.join(input_folder, file_name)
                output_file = os.path.join(output_folder, file_name)

                with open(input_file, 'r') as f:
                    input_lines = [json.loads(line) for line in f]
                    
                start_time = time.time()
                all_item = 0
                processed_item = 0
                remained_item = 0
                log(f"[START] Start processing {input_file}...", mode="INFO")
                
                output_lines = []
                for line in input_lines:
                    new_line = {}
                    for key, value in line.items():
                        try:
                            # Make results empty
                            results = []
                            
                            for idx, item in enumerate(value):
                                # Add to task queue
                                task_queue.put((idx, item))
                                
                            # wait for task queue to be empty
                            while not task_queue.empty() and _is_any_thread_runing():
                                time.sleep(1)
                                
                        except KeyboardInterrupt:
                            log("[ERROR] KeyboardInterrupt. Stopping threads...", mode="ERROR")
                            stop_event.set()
                            for thread in threads:
                                thread.join()
                            log("[ERROR] All threads stopped.", mode="ERROR")
                            log("[ERROR] Safety exit...", mode="ERROR")
                
                        # Add all remain item in queue to results
                        while not task_queue.empty():
                            idx, item = task_queue.get()
                            results.append((idx, item))
                            
                        all_item += len(results)
                        # Sort results by index and add to output_lines
                        results.sort(key=lambda x: x[0])    
                        new_line[key] = [item[1] for item in results]
                    
                    output_lines.append(new_line)
                
                with open(output_file, 'w') as f:
                    for line in output_lines:
                        if line is not None:                        
                            f.write(json.dumps(line) + '\n')
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                log(f"[SAVE FILE] Processing {processed_item}/{all_item} item, remaning {remained_item}/{all_item} in {elapsed_time:.2f} seconds in {input_file} and saved to {output_file} ")
            

        # # Convert test
        # input_file = os.path.join(input_path, dataset, f"{dataset}.test.jsonl")
        # output_file = os.path.join(output_path, dataset, f"{dataset}.test.jsonl")

        # if os.path.exists(input_file):
        #     with open(input_file, 'r') as f:
        #         input_lines = [json.loads(line) for line in f]
        
        # output_lines = [] 
        # for line in input_lines:
        #     new_line = {}
        #     for key, value in line.items():
        #         results = []
                
        #         for idx, item in enumerate(value):
        #             # Add to task queue
        #             task_queue.put((idx, item))
                    
        #         # wait for task queue to be empty
        #         while not task_queue.empty() and _is_any_thread_runing():
        #             time.sleep(1)
                    
        #         # Add all remain item in queue to results
        #         while not task_queue.empty():
        #             idx, item = task_queue.get()
        #             results.append((idx, item))
                
        #         # Sort results by index and add to output_lines
        #         results.sort(key=lambda x: x[0])    
        #         new_line[key] = [item[1] for item in results]
        #         output_lines.append(new_line)
        
        # with open(output_file, 'w') as f:
        #     for line in output_lines:
        #         if line is not None:                        
        #             f.write(json.dumps(line) + '\n')
        
        stop_event.set()
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    args = parse_arguments()
    input_root = args.input_root
    output_root = args.output_root
    datasets = args.datasets
    model = args.model
    candidate = args.candidate
    num_try = args.num_try
    logs_dir = args.logs_dir
    convert(INPUT_ROOT, OUTPUT_ROOT, DATASETS, resume=True)
