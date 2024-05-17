import json
import torch
import torchvision.transforms as transforms
from PIL import Image

# 加载模型
model_path = './hello.pth'  # 替换为你的模型路径
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
# 创建 GradCAM 实例，并指定目标层
img_tensor.requires_grad = True
cam = GradCAM(model=resnet._network, target_layer=l)
# 对模型进行前向传播并获取输出
outputs = model._network(img_tensor)['eval_logits']

cams = cam(class_idx=5,scores=outputs) # 得到热图，index是你希望看到的类别
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize, to_pil_image
# 将热图叠加到原始图像上
result = overlay_mask(pil_img, to_pil_image(cams[0].to("cpu"), mode="F"), alpha=0.5)

# 显示图像
plt.imshow(result)
plt.axis('off')
plt.title("dog")
# plt.show()
# 保存图像
plt.savefig('cams_image_dog.png')

# 关闭 Matplotlib 图形
plt.close()