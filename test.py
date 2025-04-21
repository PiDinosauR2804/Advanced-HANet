from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
text = "This is an event mention."

# Encode: raw text → token IDs
input_ids = tokenizer.encode(text, add_special_tokens=True)

# Decode lại để kiểm tra
decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)

print("Input IDs:", input_ids)
print("Decoded:", decoded_text)
