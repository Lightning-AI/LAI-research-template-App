import logging
import os
from typing import Dict, List, Optional

import lightning as L
from lit_jupyter import LitJupyter
from rich import print

from research_app.components.markdown_poster import Poster
from research_app.components.model_demo import ModelDemo
from research_app.utils import clone_repo, notebook_to_html

logger = logging.getLogger(__name__)


class StaticNotebook(L.LightningFlow):
    def __init__(self, serve_dir: str):
        super().__init__()
        self.serve_dir = serve_dir
        self.ready = True

    def configure_layout(self):
        return L.frontend.web.StaticWebFrontend(serve_dir=self.serve_dir)


class ResearchApp(L.LightningFlow):
    """Share your paper "bundled" with the arxiv link, poster, live jupyter notebook, interactive demo to try the model
    and more!

    paper: Arxiv link to your paper
    blog: Link to a blog post for your research
    github: Clone GitHub repo to the current directory.
    training_log_url: Link for experiment manager like wandb or tensorboard
    notebook_path: View a Jupyter Notebook as static html tab
    launch_jupyter_lab: Launch a full-fledged Jupyter Lab instance @Aniket might be worth mentioning somewhere that if you do then it might be subject to abuse if you share the url on the cloud
    launch_gradio: Launch Gradio demo.
        You should update the `research_app/serve/gradio_app.py` file to your use case.
    tab_order: Tabs will appear in UI in the same order as the provided list of tab names.
    """

    def __init__(
        self,
        resource_path: str,
        paper: Optional[str] = None,
        blog: Optional[str] = None,
        github: Optional[str] = None,
        notebook_path: Optional[str] = None,
        training_log_url: Optional[str] = None,
        launch_jupyter_lab: bool = False,
        launch_gradio: bool = False,
        tab_order: Optional[List[str]] = None,
    ) -> None:

        super().__init__()
        self.resource_path = os.path.abspath(resource_path)
        self.paper = paper
        self.blog = blog
        self.training_logs = training_log_url
        self.notebook_path = notebook_path
        self.launch_jupyter_lab = launch_jupyter_lab
        self.enable_gradio = launch_gradio
        self.poster = Poster(resource_path=self.resource_path)
        self.tab_order = tab_order
        if github:
            clone_repo(github)

        if launch_jupyter_lab:
            self.jupyter_lab = LitJupyter()

        if launch_gradio:
            self.model_demo = ModelDemo()
        # @Aniket what happen if i do launch_jupyterlab = true + give a notebook_path?
        if notebook_path:
            serve_dir = notebook_to_html(notebook_path)
            self.notebook = StaticNotebook(serve_dir)

    def run(self) -> None:
        if os.environ.get("TESTING_LAI"):
            print("⚡ Lightning Research App! ⚡")
        self.poster.run()
        if self.launch_jupyter_lab:
            self.jupyter_lab.run()
        if self.enable_gradio:
            self.model_demo.run()

    def configure_layout(self) -> List[Dict[str, str]]:
        tabs = []

        tabs.append({"name": "Poster", "content": self.poster.url + "/poster.html"})

        if self.blog:
            tabs.append({"name": "Blog", "content": self.blog})

        if self.paper:
            tabs.append({"name": "Paper", "content": self.paper})

        if self.launch_jupyter_lab:
            tabs.append({"name": "JupyterLab", "content": self.jupyter_lab.url})

        if self.training_logs:
            tabs.append({"name": "Training Logs", "content": self.training_logs})

        if self.enable_gradio:
            tabs.append({"name": "Model Demo", "content": self.model_demo.url})

        tabs.append({"name": "Notebook", "content": self.notebook})

        return self._order_tabs(tabs)

    def _order_tabs(self, tabs: List[dict]):
        """Reorder the tab layout."""
        if self.tab_order is None:
            return tabs
        order_int: Dict[str, int] = {e.lower(): i for i, e in enumerate(self.tab_order)}
        return sorted(tabs, key=lambda x: order_int[x["name"].lower()])


if __name__ == "__main__":
    # @Aniket maybe comment to say this is a running example, change your code here.
    resource_path = "resources"
    paper = "https://arxiv.org/pdf/2103.00020"
    blog = "https://openai.com/blog/clip/"
    github = "https://github.com/openai/CLIP"
    wandb = "https://wandb.ai/cceyda/flax-clip/runs/wlad2c2p?workspace=user-aniketmaurya"
    tabs = ["Blog", "Paper", "Poster", "Notebook", "Training Logs", "Model Demo"]

    app = L.LightningApp(
        ResearchApp(
            resource_path=resource_path,
            paper=paper,
            blog=blog,
            training_log_url=wandb,
            github=github,
            notebook_path="resources/Interacting_with_CLIP.ipynb",
            launch_jupyter_lab=False,
            launch_gradio=True,
            tab_order=tabs,
        )
    )
