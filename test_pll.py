import threading
import time
import queue
import random

task_queue = queue.Queue()
results = []
stop_event = threading.Event()
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


# Đưa 20 task vào hàng đợi
for i in range(20):
    task_queue.put((i, i))

# Thread worker
def worker(worker_id, stop_after=None):
    count = 0
    while not task_queue.empty() and not stop_event.is_set():
        try:
            idx, item = task_queue.get(timeout=1)
        except queue.Empty:
            safe_print(f"Worker {worker_id} không còn task nào để xử lý.")
            break
        
        if random.random() < 0.5:  # Giả lập một số task không được xử lý
            safe_print(f"Worker {worker_id} xử lý task {item}")
            results.append((worker_id, item+100))
        else:
            task_queue.put((idx, item))
            
        count += 1
        time.sleep(0.05)
        if stop_after and count >= stop_after:
            safe_print(f"Worker {worker_id} dừng sớm sau {count} task")
            break

if __name__ == "__main__":
    threads = []
    for i in range(3):
        # t = threading.Thread(target=worker, args=(i,), kwargs={'stop_after': 3 if i == 1 else None})
        t = threading.Thread(target=worker, args=(i,), kwargs={'stop_after': 3})
        t.start()
        threads.append(t)


    for t in threads:
        t.join()  # Join với timeout ngắn để có thể kiểm tra Ctrl+C

    while not task_queue.empty():
        try:
            idx, item = task_queue.get_nowait()
            results.append((-1, item))  # Thêm task chưa xử lý vào kết quả
        except queue.Empty:
            break
        task_queue.task_done()
    # In kết quả
    line1 = []
    line2 = []
    for res in results:
        line1.append(res[0])
        line2.append(res[1])
        
    safe_print("    Order: ", end="")
    for i in range(len(line1)):
        safe_print(f"{i:<5}", end="")
    safe_print("\nWorker ID: ", end="")
    for i in range(len(line1)):
        safe_print(f"{line1[i]:<5}", end="")
    safe_print("\n  Task ID: ", end="")
    for i in range(len(line2)):
        safe_print(f"{line2[i]:<5}", end="")

