import gradio as gr
import subprocess
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# import os
# import random
# from gradio_client import Client


subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

# Initialize Florence model
device = "cuda" if torch.cuda.is_available() else "cpu"
florence_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True).to(device).eval()
florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)

# api_key = os.getenv("HF_READ_TOKEN")

def generate_caption(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    inputs = florence_processor(text="<MORE_DETAILED_CAPTION>", images=image, return_tensors="pt").to(device)
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(
        generated_text,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(image.width, image.height)
    )
    prompt =  parsed_answer["<MORE_DETAILED_CAPTION>"]
    print("\n\nGeneration completed!:"+ prompt)
    return prompt
    # yield prompt, None
    # image_path = generate_image(prompt,random.randint(0, 4294967296))
    # yield prompt, image_path 

# def generate_image(prompt, seed=42, width=1024, height=1024):
#     try:
#         result = Client("KingNish/Realtime-FLUX", hf_token=api_key).predict(
#             prompt=prompt,
#             seed=seed,
#             width=width,
#             height=height,
#             api_name="/generate_image"
#         )
#         # Extract the image path from the result tuple
#         image_path = result[0]
#         return image_path 
#     except Exception as e:
#         raise Exception(f"Error generating image: {str(e)}")
 
io = gr.Interface(generate_caption,
                  inputs=[gr.Image(label="Input Image")],
                  outputs = [gr.Textbox(label="Output Prompt", lines=2, show_copy_button = True),
                             # gr.Image(label="Output Image")
                            ],
                  deep_link=False
                 )
io.launch(debug=True)