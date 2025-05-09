import gradio as gr
import pandas as pd
import numpy as np
import tempfile

def generate_random_csv():
    # Create a random DataFrame
    df = pd.DataFrame(np.random.rand(10, 5), columns=[f"Column {i}" for i in range(1, 6)])
    
    # Use a temporary file to save the CSV
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_file.name, index=False)
    print(f"CSV file created at: {temp_file.name}")
    return temp_file.name

# Create a Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("# Random CSV Generator")
    gr.Markdown("Click the button to generate and download a random CSV file.")
    
    # Button to trigger CSV generation
    generate_button = gr.Button("Generate CSV")
    
    # File output for downloading the CSV
    file_output = gr.File(label="Download Random CSV")
    
    # Define the interaction
    generate_button.click(fn=generate_random_csv, inputs=[], outputs=file_output)

# Launch the interface
demo.launch()