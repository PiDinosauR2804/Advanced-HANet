import json

for line in open('data/data_text/ACE/ACE.test.jsonl', 'r', encoding='utf-8'):
    ace = json.loads(line)
    
print(ace)
