import logging

import gradio as gr
from lightning.components.serve import ServeGradio
from rich.logging import RichHandler

from research_app.clip_demo import CLIPDemo

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)


class ModelDemo(ServeGradio):
    """Serve model with Gradio UI.

    You need to define i. `build_model` and ii. `predict` method and Lightning `ServeGradio` component will
    automatically launch the Gradio interface.
    """

    inputs = gr.inputs.Textbox(default="Doctor Strange Multiverse", label="Search your favourite movie here")
    outputs = gr.outputs.HTML(label="Fetch Images from <b>themoviedb.org</b>")

    def __init__(
        self,
    ):
        super(ServeGradio, self).__init__(parallel=True)
        self._model = None
        self.enable_queue = True

    def build_model(self) -> CLIPDemo:
        logger.info("loading model...")
        clip = CLIPDemo()
        logger.info("built model!")
        return clip

    def predict(self, query: str) -> str:
        return self.model.predict(query)


if __name__ == "__main__":
    demo = ModelDemo()
    demo.run()
