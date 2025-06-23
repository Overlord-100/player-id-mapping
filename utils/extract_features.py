import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

resnet = resnet18(pretrained=True)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def extract_features(cropped_img):
    img_tensor = transform(cropped_img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_tensor)
    return features.squeeze().numpy()
