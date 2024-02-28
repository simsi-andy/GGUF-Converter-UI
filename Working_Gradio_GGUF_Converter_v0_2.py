import gradio as gr
from huggingface_hub import snapshot_download
import subprocess
import sys
import os

# Function to check system requirements
def check_requirements():
    # Check if Python is installed
    python_installed = subprocess.call("python --version", shell=True) == 0
    if not python_installed:
        return "Python is not installed. Please install Python before proceeding."
    
    # Check if Git is installed
    git_installed = subprocess.call("git --version", shell=True) == 0
    if not git_installed:
        return "Git is not installed. Please install Git before proceeding."
    
    return "All system requirements are satisfied."

# Function to download and convert the model
def download_and_convert(model_id, output_name, outtype):
    # Perform system requirements check
    requirements_message = check_requirements()
    if "not installed" in requirements_message:
        return requirements_message
    
    try:
        # Download the model with Hugging Face Hub
        model_dir = snapshot_download(repo_id=model_id, cache_dir='./models')
        
        # Clone the conversion tool repo if not present
        if not os.path.isdir("llama.cpp"):
            subprocess.check_call(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"])
        
        # Install the required dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "llama.cpp/requirements.txt"])
        
        # Specify the output file name and add the .gguf extension
        outfile_path = f"./models/{output_name}.gguf"
        
        # Execute the conversion script
        subprocess.check_call([sys.executable, "llama.cpp/convert.py", model_dir, "--outfile", outfile_path, "--outtype", outtype])

        return f"Conversion was successfully completed. The file is saved as: {outfile_path}"
    except subprocess.CalledProcessError as e:
        return f"Error during conversion: {str(e)}"

# Gradio interface function
def gradio_interface(model_id, output_name, outtype):
    return download_and_convert(model_id, output_name, outtype)

# Gradio Interface setup
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Model ID:"),
        gr.Textbox(label="Output File Name (without extension):"),
        gr.Dropdown(choices=["f32", "f16", "q8_0"], label="Output Type:")
    ],
    outputs=gr.Textbox(label="Result"),
    title="Model Converter",
    description="This tool allows you to download and convert models from Hugging Face Hub. Please enter the model ID, desired output file name (without extension), and select the desired output type.",
)

iface.launch(enable_queue=False) # Disable the Flag button
