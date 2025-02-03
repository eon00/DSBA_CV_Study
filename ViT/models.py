from transformers import ViTForImageClassification, ViTConfig
import torch

def get_model(config, device, is_train=False):
    """
    ViT 모델을 생성하여 반환.
    
    - Training에서는 Pretrained 모델을 가져오거나 랜덤 초기화 모델 사용.
    - Inference에서는 학습된 모델(`use_trained_model: true`)을 불러오거나, Pretrained 여부 선택.
    
    Args:
        config (dict): 모델 설정을 담은 config.yaml
        device (torch.device): GPU 또는 CPU 설정
        is_train (bool): True이면 Training 용도, False이면 Inference 용도
    """
    model_path = config["inference"].get("model_path", None)  # 모델 가중치 파일 (Inference용)
    use_trained_model = config["inference"].get("use_trained_model", False)  # 학습된 모델 불러올지 여부
    use_pretrained = config["model"].get("pretrained", True)  # 프리트레인 모델 사용할지 여부

    if is_train:
        # ✅ Training 시에는 pretrained 여부에 따라 모델 선택
        if use_pretrained:
            print("🔹 Using pretrained ViT model for training")
            model = ViTForImageClassification.from_pretrained(
                config["model"]["base_model"], 
                num_labels=config["model"]["num_classes"],
                ignore_mismatched_sizes=True
            )
        else:
            print("🔹 Using randomly initialized ViT model for training")
            model_config = ViTConfig.from_pretrained(
                config["model"]["base_model"], 
                num_labels=config["model"]["num_classes"]
            )
            model = ViTForImageClassification(config=model_config)

    else:
        # ✅ Inference에서는 학습된 모델을 사용할지 여부 체크
        if use_trained_model and model_path:
            if use_pretrained:
                print(f"🔹 Loading pretrained & fine-tuned ViT model from {model_path}")
                model = ViTForImageClassification.from_pretrained(
                    config["model"]["base_model"], 
                    num_labels=config["model"]["num_classes"],
                    ignore_mismatched_sizes=True
                )
            else:
                print(f"🔹 Loading randomly initialized & fine-tuned ViT model from {model_path}")
                model_config = ViTConfig.from_pretrained(
                    config["model"]["base_model"], 
                    num_labels=config["model"]["num_classes"]
                )
                model = ViTForImageClassification(config=model_config)

            model.load_state_dict(torch.load(model_path, map_location=device))  # 학습된 가중치 불러오기
        
        else:
            if use_pretrained:
                print("🔹 Using pretrained ViT model for inference")
                model = ViTForImageClassification.from_pretrained(
                    config["model"]["base_model"], 
                    num_labels=config["model"]["num_classes"],
                    ignore_mismatched_sizes=True
                )
            else:
                print("🔹 Using randomly initialized ViT model for inference")
                model_config = ViTConfig.from_pretrained(
                    config["model"]["base_model"], 
                    num_labels=config["model"]["num_classes"]
                )
                model = ViTForImageClassification(config=model_config)

    return model.to(device) 
