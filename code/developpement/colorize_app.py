import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from unet import UNet

source_process = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])])

model = UNet()
model.load_state_dict(torch.load('unet.pth'))
model.eval()

def recognize_digit(image):
    img = source_process(image).unsqueeze(0)  # add a batch dimension
    with torch.no_grad():
      prediction = model(img)[0]
    save_image(prediction, "colorized.png", normalize=True)
    return "colorized.png"

gr.Interface(fn=recognize_digit, 
             inputs=gr.Image(type="pil", image_mode='L'), 
             outputs="image",
             #live=True,
             description="Select an image",
             ).launch(debug=True);