import torch

def custom_model(pretrained=False, **kwargs):
    # 모델 생성 로직 작성
    model = YourCustomModel()  # 실제 모델 클래스나 함수로 대체해야 합니다.
    if pretrained:
        # 가중치 파일 다운로드
        model_weights_url = 'https://github.com/KangMinNoh/best/raw/main/best.pt'
        state_dict = torch.hub.load_state_dict_from_url(model_weights_url, map_location=torch.device('cpu'))

        # 모델에 가중치 로드
        model.load_state_dict(state_dict)
    return model

