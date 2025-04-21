import json, os
import torch
import torch.nn.functional as F
from configs import parse_arguments
args = parse_arguments()
device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")  # type: ignore

def compute_CLLoss(Adj_mask, reprs, matsize): # compute InfoNCELoss
    logits_cl = torch.div(torch.matmul(reprs, reprs.T), args.cl_temp)
    if args.sub_max:
        logits_max_cl, _ = torch.max(logits_cl, dim=-1, keepdim=True)
        logits_cl = logits_cl - logits_max_cl
    exp_logits_cl = torch.exp(logits_cl)
    denom_cl = torch.sum(exp_logits_cl * (1 - torch.eye(matsize).to(device)), dim = -1) 
    log_prob_cl = -torch.mean((logits_cl - torch.log(denom_cl)) * Adj_mask, dim=-1)
    return torch.mean(log_prob_cl[log_prob_cl > 0])

def collect_from_json(dataset, root, split):
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
        with open(pth) as f:
            if pth.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
                if split == "train":
                    data = [list(i.values()) for i in data]
            else:
                data = json.load(f)
    return data

def sim(x, y):
    """
    Tính độ tương đồng giữa hai vectơ x, y
    
    - x: Tensor (N, D), batch của N vectơ đầu vào
    - y: Tensor (M, D), batch của M vectơ so sánh
    
    Trả về:
    - sim: Tensor (N, M), ma trận độ tương đồng giữa x và y
    """
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    
    return torch.mm(x, y.t())

@torch.no_grad()
def find_negative_labels(description_res, k=4):
    negative_dict = dict()
    description_out = {}
    description_matrix = []
    
    rel2id = dict()
    with torch.no_grad():
        for idx, (key, description) in enumerate(description_res.items()):
            rel2id[idx] = key
            description_matrix.append(description)
        
        
    description_matrix = torch.stack(description_matrix, dim=0)

    # Tính cosine similarity giữa reps và descriptions
    similarities = sim(description_matrix, description_matrix) / 5  # (N, M)
    
    # Sắp xếp theo giá trị giảm dần (dim=1 để sắp theo hàng)
    _, topk_indices = torch.topk(similarities, k=min(k+1,description_matrix.shape[0]), dim=1)  # k+1 để bỏ chính nó
    
    # Bỏ chính nó (index đầu tiên)
    topk_indices = topk_indices[:, 1:].tolist()  # Chuyển thành list để dễ dùng
    
    for i in range(len(topk_indices)):
        new_topk_indices = [rel2id[j] for j in topk_indices[i]]
        negative_dict[rel2id[i]] = new_topk_indices
    return negative_dict