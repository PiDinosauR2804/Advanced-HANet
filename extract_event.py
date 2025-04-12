from extractor.producer import Producer
from extractor.consumer import Consumer
from configs import parse_arguments
# from extractor.extractor_config import extractor_parse_arguments as parse_arguments
import json
import os
import time
import queue
from loguru import logger
from tqdm import tqdm

def get_lines_from_results(results:list)->list:
    # Sort results by line_idx, key and idx
    results.sort(key=lambda x: (x[0], x[1], x[2])) # results is a list of tuples (line_idx, key, idx, item)
    # Convert results to a list of lines
    output_lines = []
    for line_idx, key, idx, item in results:
        if line_idx >= len(output_lines):
            output_lines.append({})
        if key is not None and idx is not None:
            if key not in output_lines[line_idx]:
                output_lines[line_idx][key] = []
            output_lines[line_idx][key].append(item)
        else:
            output_lines[line_idx] = item
    return output_lines

def run(input_path:str, output_path:str, datasets:list, model:str, candidate:int, 
        num_try:int, max_consecutive_429_error:int, max_num_threads:int, resume:bool=False)->None:
    # Resume from output_path if resume is True
    if resume:
        input_path = output_path
    else:
        os.makedirs(output_path, exist_ok=True)

    # Create task queue, producer and consumer
    task_queue = queue.Queue()
    producer = Producer(task_queue)
    consumer = Consumer(task_queue, num_try=num_try, max_consecutive_429_error=max_consecutive_429_error, 
                        model=model, candidate=candidate, max_num_threads=max_num_threads)
    
    # Convert dataset
    for dataset in datasets:
        os.makedirs(os.path.join(output_path, dataset), exist_ok=True)

        for i in [0, 2, 3, 4]:
            input_folder = os.path.join(input_path, dataset, "perm"+str(i))
            if not os.path.exists(input_folder):
                logger.error(f"[ERROR] Folder {input_folder} is not exist", mode="ERROR")
                continue

            output_folder = os.path.join(output_path, dataset, "perm"+str(i))
            os.makedirs(output_folder, exist_ok=True)

            for file_name in os.listdir(input_folder):
                if not file_name.endswith(".jsonl"):
                    continue

                input_file = os.path.join(input_folder, file_name)
                output_file = os.path.join(output_folder, file_name)
                
                start_time = time.time()
                # Start producing
                producer.produce(input_file, is_train=True)
                # Start consuming
                try:
                    results, processed_item, remained_item = consumer.consume(pbar_des=f'{dataset}/perm{i}/{file_name}', model=model, candidate=candidate) # results is a list of tuples (line_idx, key, idx, item)
                except KeyboardInterrupt:
                    logger.error(f"[ERROR] KeyboardInterrupt, stop consuming safely...")
                    consumer.stop_threads()
                    
                
                # Save results to output file
                output_lines = get_lines_from_results(results) # output_lines is a list of lines
                
                with open(output_file, 'w') as f:   
                    for line in output_lines:                       
                            f.write(json.dumps(line) + '\n')
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.info(f"[SAVE FILE] Processing {processed_item}/{len(results)} item, remaning {remained_item}/{len(results)} in {elapsed_time:.2f} seconds in {input_file} and saved to {output_file} ")
            

        # Convert test
        input_file = os.path.join(input_path, dataset, f"{dataset}.test.jsonl")
        output_file = os.path.join(output_path, dataset, f"{dataset}.test.jsonl")

        if not os.path.exists(input_file):
            logger.error(f"[ERROR] File {input_file} is not exist", mode="ERROR")
            continue
        
        start_time = time.time()
        # Start producing
        producer.produce(input_file, is_train=False)
        # Start consuming
        try:
            results, processed_item, remained_item = consumer.consume(pbar_des=f'{dataset}/{dataset}/.test.jsonl', model=model, candidate=candidate) # results is a list of tuples (line_idx, key, idx, item)
        except KeyboardInterrupt:
            logger.error(f"[ERROR] KeyboardInterrupt, stop consuming safely...")
            consumer.stop_threads()
        
        # Save results to output file
        output_lines = get_lines_from_results(results) # output_lines is a list of lines
        
        with open(output_file, 'w') as f:
            for line in output_lines:
                if line is not None:                        
                    f.write(json.dumps(line) + '\n')
                    
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"[FINISHED] Processing {processed_item}/{len(results)} item, remaning {remained_item}/{len(results)} in {elapsed_time:.2f} seconds")
        logger.info(f"[SAVE FILE] All item saved to {output_file}")

if __name__ == "__main__":
    args = parse_arguments()
    input_root = args.input_root
    output_root = args.output_root
    datasets = args.datasets
    model = args.model
    candidate = args.candidate
    num_try = args.num_try
    max_consecutive_429_error = args.max_consecutive_429_error
    max_num_threads = args.max_num_threads
    resume = args.extractor_resume
    logs_dir = args.logs_dir
    
    # Configure logging
    os.makedirs(logs_dir, exist_ok=True)
    # --- Xoá handler mặc định ---
    logger.remove()
    # --- Thêm handler ghi log ra file ---
    date_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logger.add(os.path.join(logs_dir, f"{date_str}.log"), rotation="1 MB", retention="10 days", enqueue=True, level="INFO")
    # --- Thêm handler ghi log qua tqdm.write ---
    # Ghi log ra console qua tqdm.write + có màu
    logger.level("CRITICAL", color="<bg red><white>")
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level="DEBUG",
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{file: >18}: {line: <4}</cyan> - <level>{message}</level>",
    )

    run(input_root, output_root, datasets, model, candidate, num_try, max_consecutive_429_error, max_num_threads, resume)
