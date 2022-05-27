import enum
import os.path
import urllib
from typing import List

import numpy as np
import pandas as pd
from transformers import CLIPModel, CLIPProcessor, logging


def download_files():
    print("Downloading embeddings, this might take some time!")
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?export=download&id=1onKr-pfWb4l6LgL-z8WDod3NMW-nIJxE",
        "resouexamples/clip/resources/embeddings.npy",
    )
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?export=download&id=1KbwUkE0T8bpnHraqSzTeGGV4-TZO_CFB",
        "examples/clip/resources/embeddings2.npy",
    )
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?export=download&id=1bt1O-iArKuU9LGkMV1zUPTEHZk8k7L65",
        "examples/clip/resources/data.csv",
    )
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?export=download&id=19aVnFBY-Rc0-3VErF_C7PojmWpBsb5wk",
        "examples/clip/resources/data2.csv",
    )


if not os.path.exists("examples/clip/resources/data.csv"):
    download_files()

df = {0: pd.read_csv("examples/clip/resources/data.csv"), 1: pd.read_csv("examples/clip/resources/data2.csv")}
EMBEDDINGS = {
    0: np.load("examples/clip/resources/embeddings.npy"),
    1: np.load("examples/clip/resources/embeddings2.npy"),
}
for k in [0, 1]:
    EMBEDDINGS[k] = np.divide(EMBEDDINGS[k], np.sqrt(np.sum(EMBEDDINGS[k] ** 2, axis=1, keepdims=True)))
source = {0: "\nSource: Unsplash", 1: "\nSource: The Movie Database (TMDB)"}


def get_html(url_list, height=200):
    html = "<div style='margin-top: 20px; display: flex; flex-wrap: wrap; justify-content: space-evenly'>"
    for url, title, link in url_list:
        html2 = f"<img title='{title}' style='height: {height}px; margin-bottom: 10px' src='{url}'>"
        if len(link) > 0:
            html2 = f"<a href='{link}' target='_blank'>" + html2 + "</a>"
        html = html + html2
    html += "</div>"
    return html


class DATASET(enum.Enum):
    UNSPLASH = "Unsplash"
    MOVIES = "Movies"


dataset = DATASET.MOVIES.value


class CLIP:
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    def compute_text_embeddings(self, list_of_strings: List[str]):
        inputs = self.processor(text=list_of_strings, return_tensors="pt", padding=True)
        return self.model.get_text_features(**inputs)

    def image_search(self, query, n_results=24):
        text_embeddings = self.compute_text_embeddings([query]).detach().numpy()
        k = 0 if dataset == "Unsplash" else 1
        results = np.argsort((EMBEDDINGS[k] @ text_embeddings.T)[:, 0])[-1 : -n_results - 1 : -1]
        result = [(df[k].iloc[i]["path"], df[k].iloc[i]["tooltip"] + source[k], df[k].iloc[i]["link"]) for i in results]
        return result


def build_model():
    logging.get_verbosity = lambda: logging.NOTSET
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIP(processor=processor, model=model)
    return clip


def predict(model, query: str) -> str:
    results = model.image_search(query)
    return get_html(results)
