import torch
from torch import optim
from torchvision import transforms
from clip import clip
from styleclip.train import train

device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터셋 경로 설정
data_dir = "data/styleclip"

# 이미지 사이즈 설정
image_size = 512

# StyleCLIP 모델 불러오기
model, preprocess = clip.load("ViT-B/32", device=device)

# optimizer 설정
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 이미지를 로드하고 크기를 조정하는 전처리 함수
preprocess_img = transforms.Compose([
    transforms.Resize(image_size, interpolation=3),
    transforms.CenterCrop(image_size),
    transforms.ToTensor()
])

# 학습 실행
train(data_dir, model, preprocess, preprocess_img, optimizer, device, epochs=200)
