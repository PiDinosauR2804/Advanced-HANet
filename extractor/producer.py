import json
import queue
from loguru import logger
import os

class Producer:
    def __init__(self, task_queue: queue.Queue) -> None:
        self.task_queue = task_queue

    def produce(self, input_file: str, origin_input_file: str, is_train=True) -> None:
        if not input_file.endswith(".jsonl"):
            logger.error(f"[ERROR] File {input_file} is not a jsonl file")
            return
        
        if not os.path.exists(input_file):
            logger.error(f"[ERROR] File {input_file} does not exist")
            return

        logger.info(f"[PRODUCING] Start producing {input_file}...")

        with open(input_file, 'r') as f, open(origin_input_file, 'r') as f_origin:
            input_lines = [json.loads(line) for line in f]
            origin_input_lines = [json.loads(line) for line in f_origin]
            
        if len(input_lines) != len(origin_input_lines):
            logger.critical(f"[FATAL] Number of lines in input ({len(input_lines)}) and origin ({len(origin_input_lines)}) do not match")
            raise ValueError(f"Number of lines in input ({len(input_lines)}) and origin ({len(origin_input_lines)}) do not match")


        num_item = 0
        for line_idx, (line, line_origin) in enumerate(zip(input_lines, origin_input_lines)):
            if is_train:
                for key in line:
                    if key not in line_origin:
                        logger.error(f"[ERROR] Key {key} not found in origin input file")
                        raise KeyError(f"Key {key} not found in origin input file")
                    
                    value = line[key]
                    origin_value = line_origin[key]
                    
                    if len(value) != len(origin_value):
                        logger.error(f"[ERROR] Number of items in {key} ({len(value)}) and origin {key} ({len(origin_value)}) do not match")
                        raise ValueError(f"Number of items in {key} ({len(value)}) and origin {key} ({len(origin_value)}) do not match")
                    
                    for idx, (item, origin_item) in enumerate(zip(value, origin_value)):
                        item['text'] = origin_item['text'].replace("[CLS]", "").replace("[SEP]", "").lower().strip().replace(" - ", "-")
                        # item['text'] = item['text'].replace("[CLS]", "").replace("[SEP]", "").lower().strip().replace(" - ", "-")
                        self.task_queue.put((line_idx, key, idx, item))
                        num_item += 1
            else:
                self.task_queue.put((line_idx, None, None, line))
                num_item += 1

        logger.info(f"[PRODUCING] Finished producing {input_file} with {num_item} item in {len(input_lines)} lines")