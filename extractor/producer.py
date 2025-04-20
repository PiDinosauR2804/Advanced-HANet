import json
import queue
from loguru import logger
import os

class Producer:
    def __init__(self, task_queue:queue.Queue)->None:
        self.task_queue = task_queue

    def produce(self, input_file:str, is_train=True)->None:
        if not input_file.endswith(".jsonl"):
            logger.error(f"[ERROR] File {input_file} is not a jsonl file")
            return
        
        if not os.path.exists(input_file):
            logger.error(f"[ERROR] File {input_file} does not exist")
            return

        logger.info(f"[PRODUCING] Start producing {input_file}...")
        
        with open(input_file, 'r') as f:
            input_lines = [json.loads(line) for line in f]
        num_item = 0
        for line_idx, line in enumerate(input_lines):
            if is_train:
                for key, value in line.items():
                    for idx, item in enumerate(value):
                        # Add to task queue
                        # loại bỏ token [CLS] và [SEP] trong item['text']
                        item['text'] = item['text'].replace("[CLS]", "").replace("[SEP]", "").lower().strip()
                        self.task_queue.put((line_idx, key, idx, item))
                        num_item += 1
                        
            else:
                self.task_queue.put((line_idx, None, None, line))
                num_item += 1
                
        logger.info(f"[PRODUCING] Finished producing {input_file} with {num_item} item in {len(input_lines)} lines")