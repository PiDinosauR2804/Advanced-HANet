import os
import json
from loguru import logger
import copy
from utils.convert import sent2ids

def augment_data(line):
    augment_data_list = []
    for key, value in line.items():
        key_id = int(key)
        new_augment_line = {}
        new_augment_line[key] = []
        for data in value:
            if 'events' in data:
                events = data['events']
            if 'label' in data:
                labels = data['label']
            for i,label in enumerate(labels):
                if label == key:
                    event = events[i]
                    descriptions = event['description']
                    for des in descriptions:
                        new_data = {}
                        new_data['text'] = data['text']+ ' ' + des
                        new_data['events'] = [copy.deepcopy(event)]
                        if 'trigger_word' not in new_data['events']:
                            print('No trigger word in event')
                            return None
                        new_data = sent2ids(new_data)
                        new_augment_line[key].append(new_data)
                        augment_data_list.append(new_data)
                """
                for i, event in enumerate(events):
                    if 'description' in event:
                        description = event['description']
                        # Tạo dữ liệu mới bằng cách kết hợp text và description
                        #new_data = copy.deepcopy(data)
                        #new_data['text'] = new_data['text'] + ' ' + description
                        new_data = {}
                        new_data['text'] = data['text'] + description
                        new_data['events'] = [copy.deepcopy(event)]
                        new_data = sent2ids(new_data)
                        new_augment_line[key].append(new_data)
                        augment_data_list.append(new_data)
                """
    return augment_data_list

