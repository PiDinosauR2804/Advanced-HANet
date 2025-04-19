from extractor.llm import Extractor, Extractor_Gemini, is_quota_exhausted_error, is_valid_extractor
from utils.convert import sent2ids
from extractor.api_key import GEMINI_KEY
import time
import threading
import queue
from loguru import logger
from tqdm import tqdm

class Consumer:
    def __init__(self, task_queue:queue.Queue, num_try=3, max_consecutive_429_error=3, model='gemini-2.0-flash', candidate=1, max_num_threads=10)->None:
        self.task_queue = task_queue
        self.num_try = num_try
        self.max_consecutive_429_error = max_consecutive_429_error
        self.model = model
        self.candidate = candidate
        self.pbar = None
        self.results = []
        # Attribute for threading
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.print_lock = threading.Lock()
        self.append_lock = threading.Lock()
        self.processed_item = 0
        self.remained_item = 0
        self.extractors = []

        # Init extractors
        for i in range(len(GEMINI_KEY)):
            extractor = Extractor_Gemini(api_key=GEMINI_KEY[i])
            
            if is_valid_extractor(extractor):
                self.extractors.append({
                    'extractor': extractor,
                    'consecutive_429_error': 0,
                })
                
            if len(self.extractors) >= max_num_threads:
                break
        self.log(f"[INFO] Found {len(self.extractors)} valid extractors", mode="INFO")   
        self.pause_threads()
        self.threads = []
        for i in range(len(self.extractors)):
            # Create a thread for each extractor
            self.threads.append(threading.Thread(target=self.worker, args=(i,)))
            self.threads[i].start()
        self.log(f"[START] Start {len(self.threads)} threads", mode="INFO")
            
            
    def log(self, str:str, mode="INFO"):
        with self.print_lock:
            if mode == "INFO":
                logger.info(str)
            elif mode == "ERROR":
                logger.error(str)
            elif mode == "WARNING":
                logger.warning(str)
            elif mode == "DEBUG":
                logger.debug(str)
            elif mode == "FATAL":
                logger.critical(str)
            
    def stop_threads(self):
        self.stop_event.set()
        self.resume_threads()
        for thread in self.threads:
            thread.join()
            
        self.log(f"[STOP] All threads stopped", mode="INFO")
            
    def pause_threads(self):
        self.pause_event.clear()
        
        self.log(f"[PAUSE] All threads paused", mode="INFO")
        
    def resume_threads(self):
        self.pause_event.set()
        
        self.log(f"[RESUME] All threads resumed", mode="INFO")
    
    def clear_results(self):
        self.results = []
        self.processed_item = 0
        self.remained_item = 0
        
        self.log(f"[CLEAR] All results cleared", mode="INFO")
        
    def append_results(self, line_idx, key, idx, item):
        with self.append_lock:
            self.results.append((line_idx, key, idx, item))
        
    def consume_left_items(self):
        self.log(f"[ERROR] Task queue is not empty after consuming. Remaining items: {self.task_queue.qsize()}", mode="ERROR")
        # Giả sử self.task_queue là một Queue (ví dụ queue.Queue hoặc multiprocessing.Queue)
        queue_size = self.task_queue.qsize()
        pbar = tqdm(total=queue_size, desc="Consuming remaining items", unit="item")

        while not self.task_queue.empty():
            line_idx, key, idx, item = self.task_queue.get_nowait()

            # self.log(f"[INFO] Consumed unprocessed item: {item['text'][:30]}", mode="INFO")
            self.append_results(line_idx, key, idx, item)
            self.remained_item += 1
            pbar.update(1)
            pbar.close()

    def worker(self, worker_id):
        extractor = self.extractors[worker_id]
        
        while not self.stop_event.is_set():
            if extractor['consecutive_429_error'] >= self.max_consecutive_429_error:
                self.log(f"[FATAL] Worker {worker_id} got error 429 more than {self.max_consecutive_429_error:} times. Stoping ...", mode="FATAL")
                # Stop the thread
                break
            
            self.pause_event.wait()  # Wait until the pause event is set
            
            try:
                line_idx, key, idx, item = self.task_queue.get(timeout=0.1)
                self.pbar.update(1)
            except queue.Empty:
                time.sleep(1)
                continue
            
            # Check if the item is already processed
            if 'events' in item:
                # self.log(f"[SKIP] Worker {worker_id} processed item: {item['text'][:30]}")
                self.append_results(line_idx, key, idx, item)
                self.processed_item += 1
                continue
            
            # Try to process the item
            event_list = None
            for i in range(self.num_try):
                try:
                    event_list = extractor['extractor'].extract_event(item['text'], model=self.model, candidate=self.candidate)[0]
                    extractor['consecutive_429_error'] = 0
                    if event_list:
                        new_item = {
                            'text': item['text'],
                            'events': event_list,
                        }
                        new_item = sent2ids(new_item) # Add piece_ids, span and offsets
                        self.append_results(line_idx, key, idx, new_item)
                        self.processed_item += 1
                        # self.log(f"[SUCCESS] Worker {worker_id} processed item: {item['text'][:30]}", mode="INFO")
                        break
                    
                except Exception as e:
                    if is_quota_exhausted_error(e):
                        # self.log(f"[429 ERROR at ATTEMPT {i+1}/{self.num_try}] Worker {worker_id} got 429 error", mode="WARNING")
                        extractor['consecutive_429_error'] += 1
                        time.sleep(20)  # Wait for 15 seconds before retrying
                    else:
                        extractor['consecutive_429_error'] = 0
                        self.log(f"[ERROR at ATTEMPT {i+1}/{self.num_try}] Worker {worker_id} | text: {item['text']} | got error: {e}", mode="ERROR")
            else:
                if not event_list:
                    self.log(f"[ERROR] Worker {worker_id} got none event list: {item['text']}", mode="ERROR")
                else:
                    self.log(f"[ERROR] Worker {worker_id} failed to process item: {item['text']}", mode="ERROR")
                self.append_results(line_idx, key, idx, item)
                self.remained_item += 1
 
    def is_any_thread_running(self):
        return any(thread.is_alive() for thread in self.threads)
    
    def consume(self, pbar_des:str='Consuming item', model='gemini-2.0-flash', candidate=1):
        self.candidate = candidate
        self.model = model
        self.pbar = tqdm(total=self.task_queue.qsize(), desc=pbar_des, unit="item")
        self.clear_results()
        self.resume_threads()
        self.log(f"[CONSUMING] Start consuming.", mode="INFO")
        
        try:
            while not self.task_queue.empty():
                if not self.is_any_thread_running():
                    self.log(f"[ERROR] No threads are running. Stopping ...", mode="ERROR")
                    self.consume_left_items()
                    break
                
                time.sleep(1)
            self.pbar.close()
            self.pause_threads()
            self.log(f"[CONSUMING] Finished consuming.", mode="INFO")
            return self.results, self.processed_item, self.remained_item
                
        except KeyboardInterrupt:
            self.log(f"[ERROR] Consuming interrupted by user. Stopping safely...", mode="ERROR")
            self.pbar.close()
            self.stop_threads()
            self.consume_left_items()
            self.log(f"[CONSUMING] Finished consuming.", mode="INFO")
            return self.results, self.processed_item, self.remained_item