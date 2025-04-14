import json
import queue
import loguru
import os

class Producer:
    def __init__(self, task_queue:queue.Queue)->None:
        self.task_queue = task_queue

    def produce(self, input_file:str, is_train=True)->None:
        if not input_file.endswith(".jsonl"):
            loguru.logger.error(f"[ERROR] File {input_file} is not a jsonl file")
            return
        
        if not os.path.exists(input_file):
            loguru.logger.error(f"[ERROR] File {input_file} does not exist")
            return

        loguru.logger.info(f"[PRODUCING] Start producing {input_file}...")
        
        with open(input_file, 'r') as f:
            input_lines = [json.loads(line) for line in f]
        
        for line_idx, line in enumerate(input_lines):
            if is_train:
                for key, value in line.items():
                    for idx, item in enumerate(value):
                        # Add to task queue
                        self.task_queue.put((line_idx, key, idx, item))
                        
            else:
                self.task_queue.put((line_idx, None, None, line))
                
        loguru.logger.info(f"[PRODUCING] Finished producing {input_file} with {len(input_lines)} lines")