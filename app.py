import gradio as gr
from PIL import Image
import torch
from torchvision import transforms
import os
from main import Generator
def load_model(model_path, device):
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    G_AB.load_state_dict(checkpoint['G_AB'])
    G_BA.load_state_dict(checkpoint['G_BA'])
    return G_AB, G_BA

def process_image(input_image, direction, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    G_AB, G_BA = load_model(model_path, device)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Process input image
    img = transform(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if direction == "Photo to Sketch":
            output = G_AB(img)
        else:
            output = G_BA(img)
    
    # Convert output to PIL Image
    output = output.cpu().squeeze(0)
    output = output * 0.5 + 0.5  # Denormalize
    output = transforms.ToPILImage()(output)
    
    return output

def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# Photo-Sketch Converter using CycleGAN")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                model_path = gr.Textbox(label="Model Checkpoint Path", value="checkpoint_4.pth")
                direction = gr.Radio(["Photo to Sketch", "Sketch to Photo"], label="Conversion Direction", value="Photo to Sketch")
                submit_btn = gr.Button("Convert")
            
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
        
        submit_btn.click(
            fn=process_image,
            inputs=[input_image, direction, model_path],
            outputs=output_image
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
