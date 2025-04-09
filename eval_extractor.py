import os
import json
ground_truth = "data_incremental"
llm = "test"
datasets = ["MAVEN"]

def bleu_score(prediction:list[int], reference:list[int]):
    """
    Calculate BLEU score for a single prediction and reference.
    """
    # Initialize the BLEU score
    bleu = 0.0
    # Calculate the BLEU score for each n-gram (1-gram, 2-gram, etc.)
    for n in range(1, 5):
        # Create n-grams for prediction and reference
        pred_ngrams = [tuple(prediction[i:i+n]) for i in range(len(prediction)-n+1)]
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]
        # Count the number of matches between prediction and reference n-grams
        matches = sum(1 for ngram in pred_ngrams if ngram in ref_ngrams)
        # Calculate precision
        precision = matches / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
        # Calculate BLEU score using geometric mean of precisions
        bleu += precision / 4.0
    return bleu

def eval_span(gt_path:str, pr_path:str, datasets:list)->None:
    for dataset in datasets:
        # Convert for train data
        for i in range(5):
            gt_folder = os.path.join(gt_path, dataset, "perm"+str(i))
            pr_folder = os.path.join(pr_path, dataset, "perm"+str(i))

            # Kiểm tra xem thư mục có tồn tại không
            if not os.path.exists(gt_folder):
                print(f"Folder {gt_folder} does not exist!")
                continue
            
            if not os.path.exists(pr_folder):
                print(f"Folder {pr_folder} does not exist!")
                continue
            
            for file_name in os.listdir(pr_folder):
                if file_name.endswith(".jsonl"):
                    pr_file = os.path.join(pr_folder, file_name)
                    gt_file = os.path.join(gt_folder, file_name)
                    precision = 0.0
                    recall = 0.0
                    bleu = 0.0
                    count = 0
                    
                    with open(pr_file, 'r') as f1, open(gt_file, 'r') as f2:
                        if len(f1.readlines()) != len(f2.readlines()):
                            print(f"File length mismatch for\n{pr_file}\nand\n{gt_file}\npr len: {len(f1.readlines())}, gt len: {len(f2.readlines())}")
                            f1.close()
                            f2.close()
                            continue
                        f1.seek(0)  # Reset file pointer to the beginning
                        f2.seek(0)  # Reset file pointer to the beginning
                        for line1, line2 in zip(f1, f2):
                            pr_json_line = json.loads(line1)
                            gt_json_line = json.loads(line2)
                            for key in pr_json_line.keys():
                                if key not in gt_json_line:
                                    print(f"Key {key} not found in ground truth data")
                                    continue
                                pr_value = pr_json_line[key]
                                gt_value = gt_json_line[key]
                                
                                if len(pr_value) != len(gt_value):
                                    print(f"Length mismatch for key {key} in fil\n{pr_file}\nand\n{gt_file}\npr len: {len(pr_value)}, gt len: {len(gt_value)}")
                                    continue
                                for item1, item2 in zip(pr_value, gt_value):
                                    piece_ids1 = item1['piece_ids']
                                    piece_ids2 = item2['piece_ids']
                                    span1 = item1['span']
                                    span2 = item2['span']
                                    label = item2['label']
                                    # Calculate BLEU score for piece_ids
                                    bleu_piece_ids = bleu_score(piece_ids1, piece_ids2)
                                    bleu += bleu_piece_ids
                                    # Calculate precision and recall for spans
                                    for sp in span1:
                                        if sp in span2:
                                            precision += 1 / len(span1)
                                            recall += 1 / len([i for i in label if i != 0])
                                
                                    count += 1               
                            
                    # In bảng kết quả
                    print(f"Dataset: {dataset}, perm: {i}, file: {file_name}")
                    print(f"Precision: {precision/count:.4f}, Recall: {recall/count:.4f}, BLEU: {bleu/count:.4f}\n")
                    
        

eval_span(ground_truth, llm, datasets)