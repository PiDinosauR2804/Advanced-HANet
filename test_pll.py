from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    time.sleep(1)
    print(f"Done {n}")
    return n

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(task, [1, 2]))  # chỉ có 2 task → chỉ dùng 2 worker
