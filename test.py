import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, BertConfig
from torch.optim import AdamW
from transformers.models.bert.modeling_bert import BertSelfAttention

class SampleDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, max_length=32):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs

# Khởi tạo tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Dữ liệu mẫu
texts = ["Hello world!", "Deep learning is powerful.", "BERT attention mechanism"]
labels = [0, 1, 0]
dataset = SampleDataset(tokenizer, texts, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

class CustomAttention(nn.Module):
    def __init__(self, original_attention: BertSelfAttention, config):
        super(CustomAttention, self).__init__()
        self.original_attention = original_attention
        self.new_attention = BertSelfAttention(config)
        self.output_layer = nn.Linear(2 * config.hidden_size, config.hidden_size)

    def forward(self, 
                hidden_states, 
                attention_mask=None, 
                head_mask=None, 
                encoder_hidden_states=None, 
                encoder_attention_mask=None, 
                past_key_value=None, 
                output_attentions=False):
        
        # Gọi Attention gốc
        original_output = self.original_attention(
            hidden_states, 
            attention_mask, 
            head_mask, 
            encoder_hidden_states, 
            encoder_attention_mask, 
            past_key_value, 
            output_attentions
        )

        # Gọi Attention mới chạy song song
        new_output = self.new_attention(
            hidden_states, 
            attention_mask, 
            head_mask, 
            encoder_hidden_states, 
            encoder_attention_mask, 
            past_key_value, 
            output_attentions
        )

        # Lấy hidden_states từ kết quả
        original_hidden_states = original_output[0]
        new_hidden_states = new_output[0]

        # Kết hợp hai output (concatenation)
        combined_output = torch.cat((original_hidden_states, new_hidden_states), dim=-1)
        
        # Chiếu lại về kích thước ban đầu
        final_hidden_states = self.output_layer(combined_output)
        
        # Trả về cùng định dạng với BertSelfAttention:
        # (hidden_states, attention_weights, past_key_value)
        outputs = (final_hidden_states,) + original_output[1:]
        return outputs

class CustomBERTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.hidden_size, 2)  # Giả sử bài toán phân loại nhị phân

    def forward(self, 
                input_ids, 
                attention_mask=None, 
                token_type_ids=None,   # Thêm dòng này
                labels=None):
        
        # Truyền thêm token_type_ids vào BERT
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        
        # Lấy output của BERT
        pooled_output = outputs.last_hidden_state[:, 0]  # Chọn token [CLS]
        pooled_output = self.dropout(pooled_output)
        
        # Đưa qua classifier
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return logits, loss