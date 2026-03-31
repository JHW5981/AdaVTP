import re
import numpy as np
import torch


def extract_choice(text):
    """从回答中提取选项字母 (A/B/C/D等)"""
    text = str(text).strip()
    # 匹配选项模式
    patterns = [
        r'^([A-Z])[.、:：)]',  # A. 或 A、 或 A: 等
        r'[选择答案是]?\s*([A-Z])\s*[.、:：)]?$',  # 结尾的选项,
        # r'[the correct answer is:]?\s*([A-Z])\s*[.、:：)]?$',  # 结尾的选项
        r'^\s*([A-Z])\s*$',  # 单独的字母
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return text.upper() if len(text) == 1 and text.isalpha() else text


def calculate_choice_accuracy(generated, gt):
    """计算选择题准确率（提取选项后比较）"""
    correct = 0
    for g, t in zip(generated, gt):
        pred_choice = extract_choice(g)
        true_choice = extract_choice(t)
        if pred_choice == true_choice:
            correct += 1
    return correct / len(generated) if generated else 0.0

def normalize_answer(s):
    """标准化答案文本"""
    if isinstance(s, list):
        s = s[0] if s else ""
    s = str(s).lower().strip()
    # 移除标点符号
    s = re.sub(r'[^\w\s]', '', s)
    # 移除多余空格
    s = ' '.join(s.split())
    return s

def levenshtein_distance(s1, s2):
    """计算编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_relaxed_accuracy(generated, gt):
    """计算宽松准确率（答案包含在生成文本中即可）"""
    correct = 0
    for g, t in zip(generated, gt):
        g_norm = normalize_answer(g)
        # 处理ground truth是列表的情况（如VizWiz）
        if isinstance(t, list):
            t_list = [normalize_answer(ans) for ans in t]
            if any(ans in g_norm or g_norm in ans for ans in t_list if ans):
                correct += 1
        else:
            t_norm = normalize_answer(t)
            if t_norm in g_norm or g_norm in t_norm:
                correct += 1
    return correct / len(generated) if generated else 0.0

def calculate_anls(generated, gt, threshold=0.5):
    """
    计算ANLS (Average Normalized Levenshtein Similarity)
    常用于OCR和文档理解任务
    """
    def normalized_levenshtein(s1, s2):
        s1, s2 = normalize_answer(s1), normalize_answer(s2)
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        distance = levenshtein_distance(s1, s2)
        similarity = 1.0 - distance / max(len(s1), len(s2))
        return similarity if similarity >= threshold else 0.0
    
    scores = []
    for g, t in zip(generated, gt):
        if isinstance(t, list):
            # 取与任一正确答案的最高分
            score = max(normalized_levenshtein(g, ans) for ans in t)
        else:
            score = normalized_levenshtein(g, t)
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def calculate_yes_no_accuracy(generated, gt):
    """计算Yes/No问答准确率"""
    correct = 0
    for g, t in zip(generated, gt):
        g_norm = normalize_answer(g)
        t_norm = normalize_answer(t)
        
        # 提取yes/no
        g_answer = 'yes' if 'yes' in g_norm else ('no' if 'no' in g_norm else g_norm)
        t_answer = 'yes' if 'yes' in t_norm else ('no' if 'no' in t_norm else t_norm)
        
        if g_answer == t_answer:
            correct += 1
    
    return correct / len(generated) if generated else 0.0




def select_hidden_states(hidden_states: torch.Tensor, mask: torch.Tensor):
    """
    根据 mask 从 hidden_states 中筛选，保持 batch 维度。仅支持 batch=1。
    hidden_states: (1, T, D), mask: (1, T), 1 表示保留
    返回: (1, T', D)
    """
    indices = mask[0].nonzero(as_tuple=True)[0]
    return hidden_states[0, indices, :].unsqueeze(0)


def select_position_ids(position_ids: torch.Tensor, mask: torch.Tensor):
    """
    根据 mask 从 position_ids 中筛选，保持 batch 维度。仅支持 batch=1。
    position_ids: (1, T) 或 (3, 1, T), mask: (1, T)
    返回: (1, T') 或 (3, 1, T')
    """
    indices = mask[0].nonzero(as_tuple=True)[0]
    if position_ids.dim() == 3:
        return position_ids[:, 0, indices].unsqueeze(1)
    return position_ids[0, indices].unsqueeze(0)

def select_causal_mask(causal_mask: torch.Tensor, mask: torch.Tensor):
    """
    根据 mask 从 causal_mask 中筛选，保持 batch 维度。仅支持 batch=1。
    支持两种形状：
    - causal_mask: (1, 1, T, T)，第二维为 head 广播维，mask: (1, T)，返回: (1, 1, T', T')
    - causal_mask: (1, 1, 1, T)，mask: (1, T)，返回: (1, 1, 1, T')
    """
    indices = mask[0].nonzero(as_tuple=True)[0]
    if causal_mask.dim() == 4 and causal_mask.shape[2] == 1:
        # (1, 1, 1, T) -> (1, 1, 1, T')
        return causal_mask[:, :, :, indices]
    # (1, 1, T, T) -> (1, 1, T', T')
    return causal_mask[0, :, indices, :][:, :, indices].unsqueeze(0)

def select_position_embeddings(position_embeddings: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor):
    """
    根据 mask 从 position_embeddings 中筛选，保持 batch 维度。仅支持 batch=1。
    position_embeddings: (cos, sin)，每个形状为 (3, 1, T, D), mask: (1, T)
    返回: (cos_selected, sin_selected)，每个形状为 (3, 1, T', D)
    """
    cos, sin = position_embeddings
    indices = mask[0].nonzero(as_tuple=True)[0]
    return cos[:, :, indices, :], sin[:, :, indices, :]

