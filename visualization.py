import json
import torch
import torchvision.transforms as transforms
from PIL import Image


model_path = './hello.pth'  
model = torch.load(model_path)
resnet = model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_path = './n02091467_282.JPEG'
pil_img = Image.open(img_path, mode="r").convert("RGB")
img_tensor = transform(pil_img).unsqueeze(0).to(device)
print(img_tensor.shape)

from torchcam.methods import GradCAM
l = resnet._network.convnets[1].stage_3[3]

img_tensor.requires_grad = True
cam = GradCAM(model=resnet._network, target_layer=l)

outputs = model._network(img_tensor)['eval_logits']

cams = cam(class_idx=5,scores=outputs) 
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize, to_pil_image

result = overlay_mask(pil_img, to_pil_image(cams[0].to("cpu"), mode="F"), alpha=0.5)


plt.imshow(result)
plt.axis('off')
plt.title("dog")
# plt.show()

plt.savefig('cams_image_dog.png')


plt.close()