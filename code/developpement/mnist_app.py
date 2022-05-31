import gradio as gr
import torch
from models import MNISTNet



model = MNISTNet()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

def recognize_digit(image):
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.  # add a batch dimension
    prediction = model(image)[0]
    probabilities = torch.nn.functional.softmax(prediction, dim=0)
    return {str(i): probabilities[i].item() for i in range(10)}

gr.Interface(fn=recognize_digit, 
             inputs="sketchpad", 
             outputs=gr.outputs.Label(num_top_classes=3),
             live=True,
             description="Draw a number on the sketchpad to see the model's prediction.",
             ).launch(debug=True);