import logging

import gradio as gr
from lightning.app.components.serve import ServeGradio
from rich.logging import RichHandler

from ..model import Model

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)

source_image_path = './assets/source.png'
reference_video_path = './assets/driving.mp4'


class ModelDemo(ServeGradio):
    """Serve model with Gradio UI.

    You need to define i. `build_model` and ii. `predict` method and Lightning `ServeGradio` component will
    automatically launch the Gradio interface.
    """

    inputs = [gr.inputs.Textbox(), gr.inputs.Textbox()]
    outputs = gr.outputs.HTML(label="Images from Unsplash")
    enable_queue = True
    examples = [[source_image_path, reference_video_path]]

    def __init__(self):
        super().__init__(parallel=True)

    def build_model(self) -> Model:
        logger.info("loading model...")
        model = Model()
        logger.info("built model!")
        return model

    def predict(self, source_image, reference_video) -> str:
        return self.model.predict(source_image, reference_video)
