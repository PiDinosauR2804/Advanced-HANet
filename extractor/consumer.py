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
        self.processed_item = 0
        self.remained_item = 0
        self.queue_waiting_time = 1
        self.error_waiting_time = 20
        self.extractors = []
        # Attribute for threading
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.lock = threading.Lock()

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
        logger.info(f"[INFO] Found {len(self.extractors)} valid extractors")   
        self.pause_threads()
        self.threads = []
        for i in range(len(self.extractors)):
            # Create a thread for each extractor
            self.threads.append(threading.Thread(target=self.worker, args=(i,)))
            self.threads[i].start()
        logger.info(f"[START] Start {len(self.threads)} threads")
            
    def stop_threads(self):
        self.stop_event.set()
        self.resume_threads()
        for thread in self.threads:
            thread.join()
            
        logger.info(f"[STOP] All threads stopped")
            
    def pause_threads(self):
        self.pause_event.clear()
        time.sleep(2*self.queue_waiting_time)  # Wait for a while before stopping the threads
        
        logger.info(f"[PAUSE] All threads paused")
        
    def resume_threads(self):
        self.pause_event.set()
        
        logger.info(f"[RESUME] All threads resumed")
    
    def clear_results(self):
        self.results = []
        self.processed_item = 0
        self.remained_item = 0
        
        logger.info(f"[CLEAR] All results cleared")
        
    def append_processed_item(self, line_idx, key, idx, item):
        with self.lock:
            self.results.append((line_idx, key, idx, item))
            self.processed_item += 1
            self.task_queue.task_done()
            self.pbar.update(1)
        
    def append_remained_item(self, line_idx, key, idx, item):
        with self.lock:
            self.results.append((line_idx, key, idx, item))
            self.remained_item += 1
            self.task_queue.task_done()
            self.pbar.update(1)
        
    def consume_left_items(self):
        while not self.task_queue.empty():
            line_idx, key, idx, item = self.task_queue.get_nowait()
            self.append_remained_item(line_idx, key, idx, item)

    def worker(self, worker_id):
        extractor = self.extractors[worker_id]
        
        while not self.stop_event.is_set():
            if extractor['consecutive_429_error'] >= self.max_consecutive_429_error:
                logger.critical(f"[FATAL] Worker {worker_id} got error 429 more than {self.max_consecutive_429_error:} times. Stoping ...")
                # Stop the thread
                break
            
            self.pause_event.wait()  # Wait until the pause event is set
            
            try:
                line_idx, key, idx, item = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                time.sleep(self.queue_waiting_time)  # Wait for a while before checking the queue again
                continue
            
            # Check if the item is already processed
            if 'events' in item:
                self.append_processed_item(line_idx, key, idx, item)
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
                        self.append_processed_item(line_idx, key, idx, new_item)
                        break
                    
                except Exception as e:
                    if is_quota_exhausted_error(e):
                        extractor['consecutive_429_error'] += 1
                        time.sleep(self.error_waiting_time)  # Wait for 15 seconds before retrying
                    else:
                        extractor['consecutive_429_error'] = 0
                        logger.error(f"[ERROR at ATTEMPT {i+1}/{self.num_try}] Worker {worker_id} got error: {e} | Text: {item['text']}")
            else:
                if not event_list:
                    logger.error(f"[FAIL] Worker {worker_id} cannot extract event list after {self.num_try} attempts. | Text: {item['text']}")
                else:
                    logger.error(f"[FAIL] Worker {worker_id} fail to process event list: {event_list} | Text: {item['text']}")
                self.append_remained_item(line_idx, key, idx, item)
 
    def is_any_thread_running(self):
        return any(thread.is_alive() for thread in self.threads)
    
    def consume(self, pbar_des:str='Consuming item', model='gemini-2.0-flash', candidate=1):
        self.candidate = candidate
        self.model = model
        self.pbar = tqdm(total=self.task_queue.qsize(), desc=pbar_des, unit="item")
        self.clear_results()
        logger.info(f"[CONSUMING] Start consuming.")
        self.resume_threads()
        
        try:
            while self.task_queue.unfinished_tasks > 0:
                if not self.is_any_thread_running():
                    logger.critical(f"[FATAL] No threads are running. Consuming remained {self.task_queue.qsize()} items ...")
                    self.consume_left_items()
                    break
                
                time.sleep(1)
            
            self.pause_threads()
            
        except KeyboardInterrupt:
            logger.critical(f"[FALTAL] Consuming interrupted by user. Stopping safely...")
            self.stop_threads()
            self.consume_left_items()
            
        finally:
            self.pbar.close()
            logger.info(f"[CONSUMING] Finished consuming.")
            return self.results, self.processed_item, self.remained_item