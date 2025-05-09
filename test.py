from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from configs import parse_arguments
from utils.tools import collect_from_json
import json, os

import numpy as np
args = parse_arguments()

def collect_from_json(dataset, root, split):
    key = None
    default = ['train', 'dev', 'test']
    if split == "train":
        pth = os.path.join(root, dataset, "perm"+str(args.perm_id), f"{dataset}_{args.task_num}task_{args.class_num // args.task_num}way_{args.shot_num}shot.{split}.jsonl")
    elif split in ['dev', 'test']:
        pth = os.path.join(root, dataset, f"{dataset}.{split}.jsonl")
    elif split == "stream":
        pth = os.path.join(root, dataset, f"stream_label_{args.task_num}task_{args.class_num // args.task_num}way.json")
    else:
        raise ValueError(f"Split \"{split}\" value wrong!")
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Path {pth} do not exist!")
    else:
        print(f"Opening path: {pth}")
        with open(pth) as f:
            if pth.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
                if split == "train":
                    key = [list(i.keys()) for i in data]
                    data = [list(i.values()) for i in data]
                    
            else:
                data = json.load(f)
    return data, key

class DescriptionDataset(Dataset):
    def __init__(self, args, tokenizer, learned_types):
        file_path_description = f"description_data/{args.dataset}/description_trigger_dict.json"
        with open(file_path_description, 'r', encoding='utf-8') as f:
            data_description = json.load(f)
        
        self.data = []
        self.max_seqlen = args.max_seqlen
        self.num_description = args.num_description
        
        for key, value in data_description.items():
            if int(key) not in learned_types:
                continue
            for idx, sample in enumerate(value):
                if idx < self.num_description:
                    input_ids = tokenizer.encode(sample, add_special_tokens=True)

                    # truncate hoặc pad token
                    if len(input_ids) >= self.max_seqlen + 2:
                        token_sep = input_ids[-1]
                        token = input_ids[:self.max_seqlen + 1] + [token_sep]
                    else:
                        token = input_ids + [0] * (self.max_seqlen + 2 - len(input_ids))
                    
                    token_mask = [1 if tkn != 0 else 0 for tkn in token]

                    self.data.append((
                        token,                 # input_ids
                        token_mask,            # mask
                        int(key), 
                    ))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class MAVEN_Dataset(Dataset):
    def __init__(self, tokens, labels, masks, spans) -> None:
        super(Dataset).__init__()
        self.tokens = tokens
        self.masks = masks
        self.labels = labels
        self.spans = spans
        # self.requires_cl = [0 for _ in range(len(spans))]
        # self.labels = []
        # for stream in streams:
        #     for lb in stream:
        #         if not lb in self.label2idx:
        #             self.label2idx[lb] = len(self.label2idx)
        # for label_ls in labels:
        #     self.labels.append([self.label2idx[label]  for label in label_ls])
    def __getitem__(self, index):
        return [self.tokens[index], self.labels[index], self.masks[index], self.spans[index]]
    def __len__(self):
        return len(self.labels)
    def extend(self, tokens, labels, masks, spans):
        self.tokens.extend(tokens)
        self.labels.extend(labels)
        self.masks.extend(masks)
        self.spans.extend(spans)
        # self.requires_cl.extend(requires_cl)
    # def collate_fn(self, batch):
    #     batch = pad_sequence([torch.LongTensor(item) for item in batch[2]])
    #     pass

def collect_dataset(dataset, root, split, label2idx, stage_id, labels):
    if split == 'train':
        data = [instance for t in collect_from_json(dataset, root, split)[stage_id] for instance in t]
    else:
        data = collect_from_json(dataset, root, split)
    data_tokens, data_labels, data_masks, data_spans = [], [], [], []
    for dt in tqdm(data):
        # pop useless properties
        if 'mention_id' in dt.keys():
            dt.pop('mention_id')
        if 'sentence_id' in dt.keys():    
            dt.pop('sentence_id')
        # if split == 'train':
        add_label = []
        add_span = []
        new_t = {}
        for i in range(len(dt['label'])):
            if dt['label'][i] in labels or dt['label'][i] == 0: # if the label of instance is in the query
                add_label.append(dt['label'][i]) # append the instance and the label
                add_span.append(dt['span'][i]) # the same as label
        if len(add_label) != 0:
            token = dt['piece_ids']
            new_t['label'] = add_label
            valid_span = add_span
            valid_label = [label2idx[item] if item in label2idx else 0 for item in add_label]
        # else:
        #     token = dt['piece_ids']
        #     valid_span = dt['span'].copy()
        #     valid_label = [label2idx[item] if item in label2idx else 0 for item in dt['label']]
            # max_seqlen = 90
        max_seqlen = args.max_seqlen # 344, 249, 230, 186, 167
        if len(token) >= max_seqlen + 2:
            token_sep = token[-1]
            token = token[:max_seqlen + 1] + [token_sep]
            invalid_span = np.unique(np.nonzero(np.asarray(valid_span) > max_seqlen)[0])
            invalid_span = invalid_span[::-1]
            for invalid_idx in invalid_span:
                valid_span.pop(invalid_idx)
                valid_label.pop(invalid_idx)
        if len(token) < max_seqlen + 2:
            token = token + [0] * (max_seqlen + 2 - len(token))
        token_mask = [1 if tkn != 0 else 0 for tkn in token]
            # span_mask = []
            # for i in range(len(token)):
            #     span_mask.append([0, 0])
            # for item in valid_span:
            #     for i in range(len(item)):
            #         span_mask[item[i]][i] = 1
        data_tokens.append(token)
        data_labels.append(valid_label)
        data_masks.append(token_mask)
        data_spans.append(valid_span)
            # data_spans.append(valid_span)
    if args.my_test:
        return MAVEN_Dataset(data_tokens[:100], data_labels[:100], data_masks[:100], data_spans[:100]) # TODO: deprecated, used for debugging, not for test!
    else:
        return MAVEN_Dataset(data_tokens, data_labels, data_masks, data_spans)

def collect_exemplar_dataset(dataset, root, split, label2idx, stage_id, labels):
    data = [[instance for instance in t] for t in collect_from_json(dataset, root, split)[stage_id]]
    data_tokens, data_labels, data_masks, data_spans = [], [], [], []
    for idx, task_data in enumerate(tqdm(data)):
        for dt in task_data:
            # pop useless properties
            if 'mention_id' in dt.keys():
                dt.pop('mention_id')
            if 'sentence_id' in dt.keys():    
                dt.pop('sentence_id')
            # if split == 'train':
            add_label = []
            add_span = []
            new_t = {}
            for i in range(len(dt['label'])):
                if dt['label'][i] == labels[idx]: 
                    add_label.append(dt['label'][i]) 
                    add_span.append(dt['span'][i])
            if len(add_label) != 0:
                token = dt['piece_ids']
                new_t['label'] = add_label
                valid_span = add_span
                valid_label = [label2idx[item] if item in label2idx else 0 for item in add_label]
            # else:
            #     token = dt['piece_ids']
            #     valid_span = dt['span'].copy()
            #     valid_label = [label2idx[item] if item in label2idx else 0 for item in dt['label']]
                # max_seqlen = 90
            max_seqlen = args.max_seqlen # 344, 249, 230, 186, 167
            if len(token) >= max_seqlen + 2:
                token_sep = token[-1]
                token = token[:max_seqlen + 1] + [token_sep]
                invalid_span = np.unique(np.nonzero(np.asarray(valid_span) > max_seqlen)[0])
                invalid_span = invalid_span[::-1]
                for invalid_idx in invalid_span:
                    valid_span.pop(invalid_idx)
                    valid_label.pop(invalid_idx)
            if len(token) < max_seqlen + 2:
                token = token + [0] * (max_seqlen + 2 - len(token))
            token_mask = [1 if tkn != 0 else 0 for tkn in token]
                # span_mask = []
                # for i in range(len(token)):
                #     span_mask.append([0, 0])
                # for item in valid_span:
                #     for i in range(len(item)):
                #         span_mask[item[i]][i] = 1
            data_tokens.append(token)
            data_labels.append(valid_label)
            data_masks.append(token_mask)
            data_spans.append(valid_span)
    return MAVEN_Dataset(data_tokens, data_labels, data_masks, data_spans)
                
def collect_sldataset(dataset, root, split, label2idx, stage_id, labels):
    data_stream , key_stream = collect_from_json(dataset, root, split)
    new_labels = []
    for i in key_stream[stage_id]:
        if int(i) in labels:
            new_labels.append(int(i))
    data = [[instance for instance in t] for t in data_stream[stage_id]]
    data_tokens, data_labels, data_masks, data_spans = [], [], [], []
    for idx, task_data in enumerate(tqdm(data)):
        for dt in task_data:
            # pop useless properties
            if 'mention_id' in dt.keys():
                dt.pop('mention_id')
            if 'sentence_id' in dt.keys():    
                dt.pop('sentence_id')
            # if split == 'train':
            add_label = []
            add_span = []
            new_t = {}
            print('----------------')
            print(dt['label'])
            for i in range(len(dt['label'])):
                if dt['label'][i] == new_labels[idx] or dt['label'][i] == 0: 
                    add_label.append(dt['label'][i]) 
                    add_span.append(dt['span'][i])
            if len(add_label) != 0:
                token = dt['piece_ids']
                new_t['label'] = add_label
                valid_span = add_span
                valid_label = [label2idx[item] if item in label2idx else 0 for item in add_label]
            print(add_label)
            # else:
            #     token = dt['piece_ids']
            #     valid_span = dt['span'].copy()
            #     valid_label = [label2idx[item] if item in label2idx else 0 for item in dt['label']]
                # max_seqlen = 90
            max_seqlen = args.max_seqlen # 344, 249, 230, 186, 167
            if len(token) >= max_seqlen + 2:
                token_sep = token[-1]
                token = token[:max_seqlen + 1] + [token_sep]
                invalid_span = np.unique(np.nonzero(np.asarray(valid_span) > max_seqlen)[0])
                invalid_span = invalid_span[::-1]
                for invalid_idx in invalid_span:
                    valid_span.pop(invalid_idx)
                    valid_label.pop(invalid_idx)
            if len(token) < max_seqlen + 2:
                token = token + [0] * (max_seqlen + 2 - len(token))
            token_mask = [1 if tkn != 0 else 0 for tkn in token]

                # span_mask = []
                # for i in range(len(token)):
                #     span_mask.append([0, 0])
                # for item in valid_span:
                #     for i in range(len(item)):
                #         span_mask[item[i]][i] = 1
            data_tokens.append(token)
            data_labels.append(valid_label)
            data_masks.append(token_mask)
            data_spans.append(valid_span)
    if args.my_test:
        return MAVEN_Dataset(data_tokens[:100], data_labels[:100], data_masks[:100], data_spans[:100]) # TODO:test use
    else:
        return MAVEN_Dataset(data_tokens, data_labels, data_masks, data_spans)

def collect_eval_sldataset(dataset, root, split, label2idx, stage_id, labels):
    data = collect_from_json(dataset, root, split)
    data_tokens, data_labels, data_masks, data_spans = [], [], [], []
    for dt in tqdm(data):
        # pop useless properties
        if 'mention_id' in dt.keys():
            dt.pop('mention_id')
        if 'sentence_id' in dt.keys():    
            dt.pop('sentence_id')
        # if split == 'train':
        add_label = []
        add_span = []
        new_t = {}
        for i in range(len(dt['label'])):
            if dt['label'][i] in labels or dt['label'][i] == 0: # if the label of instance is in the query
                add_label.append(dt['label'][i]) # append the instance and the label
                add_span.append(dt['span'][i]) # the same as label
        if len(add_label) != 0:
            token = dt['piece_ids']
            new_t['label'] = add_label
            valid_span = add_span
            valid_label = [label2idx[item] if item in label2idx else 0 for item in add_label]
        # else:
        #     token = dt['piece_ids']
        #     valid_span = dt['span'].copy()
        #     valid_label = [label2idx[item] if item in label2idx else 0 for item in dt['label']]
            # max_seqlen = 90
        max_seqlen = args.max_seqlen # 344, 249, 230, 186, 167
        if len(token) > max_seqlen + 2:
            token_sep = token[-1]
            token = token[:max_seqlen + 1] + [token_sep]
            invalid_span = np.unique(np.nonzero(np.asarray(valid_span) > max_seqlen)[0])
            invalid_span = invalid_span[::-1]
            for invalid_idx in invalid_span:
                valid_span.pop(invalid_idx)
                valid_label.pop(invalid_idx)
        if len(token) < max_seqlen + 2:
            token = token + [0] * (max_seqlen + 2 - len(token))
        token_mask = [1 if tkn != 0 else 0 for tkn in token]
            # span_mask = []
            # for i in range(len(token)):
            #     span_mask.append([0, 0])
            # for item in valid_span:
            #     for i in range(len(item)):
            #         span_mask[item[i]][i] = 1
        data_tokens.append(token)
        data_labels.append(valid_label)
        data_masks.append(token_mask)
        data_spans.append(valid_span)
            # data_spans.append(valid_span)
    if args.my_test:
        return MAVEN_Dataset(data_tokens[:100], data_labels[:100], data_masks[:100], data_spans[:100]) # TODO:test use
    else:
        return MAVEN_Dataset(data_tokens, data_labels, data_masks, data_spans)

label2idx = {0: 0, 9: 1, 2: 2, 6: 3, 3: 4, 1: 5, 5: 6, 7: 7, 10: 8, 8: 9, 4: 10}
stage = 0
labels = [9, 2]

# a = collect_sldataset('ACE', 'data\data_ids', "train", label2idx, stage, labels)
# print(a[0][1])

a = collect_sldataset('ACE', 'data\data_ids_enhence', "train", label2idx, stage, labels)
print(a[0][1])
print(a[1][1])
print(a[2][1])
print(a[3][1])
print(a[4][1])
print(a[5][1])