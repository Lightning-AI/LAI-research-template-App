import logging

from lightning import LightningWork
from mkposters import mkposter

logger = logging.getLogger(__name__)


class Poster(LightningWork):
    def __init__(
        self,
        resource_path: str,
        code_style="github",
        background_color="#F6F6EF",
    ):
        # @Aniket docstring?
        super().__init__(parallel=True)
        self.resource_path = resource_path
        self.code_style = code_style
        self.background_color = background_color
        self.ready = False

    def run(self):
        self.ready = True
        mkposter(
            datadir=self.resource_path,
            background_color=self.background_color,
            code_style=self.code_style,
            port=self.port,
        )
