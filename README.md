# Lightning Research Poster Template

Use this app to share your research paper results. This app lets you connect a blogpost, arxiv paper, and jupyter notebook and even have an interactive demo for people to play with the model. This app also allows industry practitioners to productionize your work by adding inference components (sub 1ms inference time), data pipelines, etc.

Research Poster App helps Authors and Readers to publish and view research, code, experiment reports,
articles or any resource within the same app.

## Get started

You can fork or clone this template app and start editing or overriding the content.

### Installation

```
git clone https://github.com/PyTorchLightning/lightning-template-research-app.git
cd lightning-template-research-app
pip install -r requirements.txt
pip install -e .
```

### Share Research with Lightning App

The poster app has a `ResearchApp` class that provides flags that you can use to quickly build an app without knowing
any web development.

You can provide the links for each flag, and the app will automatically load and show each of the content in tabs.

You can fork and clone the repo to edit the arguments and update the markdown poster or Gradio app.

```python
# update app.py at the root of the repo

paper = "https://arxiv.org/pdf/2103.00020.pdf"
blog = "https://openai.com/blog/clip/"
github = "https://github.com/mlfoundations/open_clip"
wandb = "https://wandb.ai/aniketmaurya/herbarium-2022/runs/2dvwrme5"

app = LightningApp(
    ResearchApp(
        paper=paper,
        blog=blog,
        experiment_manager=wandb,
        enable_notebook=True,
        enable_gradio=True,
    )
)
```

To run this app, launch the terminal and enter `lightning run app FILENAME.py`

You should see something like this in your browser:

![image](./assets/demo.png)

You can also run the app on cloud by just providing `cloud` flag in the command.
`lightning run app app.py --name my_research_app --cloud`

Here is a quick video walk-through of this app -

## Contributions

**Step 1:** Install the `pre-commit` hook `pre-commit install`.

**Step 2:** Create a new branch with your code changes.

(Optional) Run pre-commit locally to check for any errors before committing: `pre-commit run --all-files`.

**Step 3:** Submit a pull request to the `lightning-template-research-app` main branch repository.
