
## Installation
```bash
# clone the repository
git clone https://github.com/mackelab/labproject.git

# (optional but recommended) create conda environment
conda create -n labproject python=3.9
conda activate labproject

# install labproject package with dependencies
pip install -e ".[dev,docs]"

# install pre-commit hooks for black auto-formatting
pre-commit install
```

`pip install -e .` installs the labproject package in editable mode, i.e. changes to the code are immediately reflected in the package.

The environment now contains, `numpy`, `scipy`, `matplotlib`, `torch`, and `jupyter`.

## Development

Develop code in your desired way, e.g. in local notebooks (you don't commit, the public notebooks for the website can be found in `docs/notebooks/`). 

Once you want other people to see your figure/experiment, add them to `plotting.py` and `experiments.py` and call the corresponding functions in `run.py`. 

After committing and pushing your changes, GitHub will execute every `run_*.py` and update the figures in Overleaf. 

You can obviously also run it yourself with `python labproject/run.py`.



## Documentation

We use mkdocs to create the public version of our tutorials (notebooks) and the API documentation. mkdocs are written in Markdown and are found in `docs/`.

After installing the necessary dependencies with `pip install -e ".[docs]"`, you can view your local version with `mkdocs serve` in `labproject/` and then open http://127.0.0.1:8000/.

To add your notebooks as pages in the docs, add the following to `mkdocs.yml`:
```yaml
pages:
  - Notebooks:
    - 'notebooks/your_notebook.ipynb' # this part is your new notebook
```

To add your functions to the API documentation, add the following to `docs/api.md`:
```markdown
### Your module
::: labproject.your_module
```
Then run `mkdocs build` to update the documentation (or `mkdocs serve` to view it locally).

For adding new pages, create a new markdown file in `docs/` and add it to `mkdocs.yml`:
```yaml
pages:
  - 'your_new_page.md'
```

### Notebooks
The jupyter notebooks are found in `docs/notebooks/`.

For your convenience, at the beginning of the jupyter notebook, run    
```python
%load_ext autoreload
%autoreload 2
```
for automatic reloading of modules, in case you make any running changes to the labproject code.
