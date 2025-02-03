import numpy as np
import pandas as pd

def accuracy(y_true, y_pred):
    """정확도 (Accuracy) 계산"""
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, average='macro'):
    """정밀도 (Precision) 계산"""
    unique_classes = np.unique(y_true)
    precisions = []
    
    for cls in unique_classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        if tp + fp == 0:
            precisions.append(0)
        else:
            precisions.append(tp / (tp + fp))
    
    return np.mean(precisions) if average == 'macro' else np.sum(precisions) / len(y_true)

def recall(y_true, y_pred, average='macro'):
    """재현율 (Recall) 계산"""
    unique_classes = np.unique(y_true)
    recalls = []
    
    for cls in unique_classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        if tp + fn == 0:
            recalls.append(0)
        else:
            recalls.append(tp / (tp + fn))
    
    return np.mean(recalls) if average == 'macro' else np.sum(recalls) / len(y_true)

def f1_score(y_true, y_pred, average='macro'):
    """F1 점수 계산"""
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    
    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)

# CSV 파일 불러오기
def load_data(file_path):
    """CSV 파일에서 데이터 로드"""
    df = pd.read_csv(file_path)
    return df['True Label'].values, df['Predicted Label'].values  # 실제값, 예측값 반환


# 테스트 실행
if __name__ == "__main__":
    augmentation_or_not = "augmentation"
    # augmentation_or_not = "non_augmentation"

    
    # pretrained_or_not = "non_pretrained"
    pretrained_or_not = "False"
    # pretrained_or_not = "True"

    

    # print(f"{augmentation_or_not}_Pretrained_{pretrained_or_not}")
    # file_path = f"../result/inference_results_{augmentation_or_not}_{pretrained_or_not}.csv"  # 파일 경로 설정
    
    print(f"not_pretrained_not_finetuning")
    file_path = f"../result/results_scratch.csv"  # 파일 경로 설정
    
    y_true, y_pred = load_data(file_path)

    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")