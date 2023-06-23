import torch
import requests

# 가중치 파일을 다운로드하여 로드하는 함수
def load_weights(model, model_weights_url):
    response = requests.get(model_weights_url)
    response.raise_for_status()
    with open('best.pt', 'wb') as f:
        f.write(response.content)
    state_dict = torch.load('best.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

# 호출 가능한 함수로 가중치를 로드하는 예시 함수
def yolov7tiny(pretrained=False, **kwargs):
    model = YOLOv7Tiny()  # 실제 모델 클래스나 함수로 대체해야 합니다.
    if pretrained:
        model_weights_url = 'https://github.com/KangMinNoh/best/best.pt'
        model = load_weights(model, model_weights_url)
    return model

