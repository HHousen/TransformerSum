import glob
import os
import sys

import gradio as gr

sys.path.insert(0, os.path.abspath("./src"))
from extractive import ExtractiveSummarizer  # noqa: E402


def summarize_text(text, model_choice):
    summarizer = ExtractiveSummarizer.load_from_checkpoint(model_choice, strict=False)
    return summarizer.predict(text)


model_choices = glob.glob("./models/*.ckpt")
model_options = gr.inputs.Dropdown(model_choices, label="ML Model")
output_text = gr.outputs.Textbox()
gr.Interface(summarize_text, ["textbox", model_options], output_text).launch()
