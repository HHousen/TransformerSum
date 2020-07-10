import glob
import gradio as gr
from extractive import ExtractiveSummarizer

def summarize_text(text, model_choice):
    summarizer = ExtractiveSummarizer.load_from_checkpoint(model_choice)
    return summarizer.predict(text)

model_choices = glob.glob("./models/*.ckpt")
model_options = gr.inputs.Dropdown(model_choices, label="ML Model")
output_text = gr.outputs.Textbox()
gr.Interface(summarize_text, ["textbox", model_options], output_text).launch()