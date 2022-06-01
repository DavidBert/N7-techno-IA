import argparse
import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
source_process = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])])

def recognize_digit(image):
    image = source_process(image).unsqueeze(0)  # add a batch dimension
    with torch.no_grad():
      prediction = model(image.to(device))[0]
    save_image(prediction, "colorized.png", normalize=True)
    return "colorized.png"

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, default = 'unet.pth', help='weights path')
    args = parser.parse_args()

    model = UNet().to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()
    gr.Interface(fn=recognize_digit, 
                inputs=gr.Image(type="pil", image_mode='L'), 
                outputs="image",
                #live=True,
                description="Select an image",
                ).launch(debug=True, share=True);