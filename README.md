
## Installation

Please execute the following commands in your terminal to install the labproject package and its dependencies. We recommend using a conda environment to avoid conflicts with other packages.

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

Develop code in your desired way, e.g. in local notebooks (you don't commit them, they will be automatically ignored). The public notebooks i.e. for the website can be found in `docs/notebooks/`, if you want to share the whole notebook then move it in this directory.

### Metrics
If you implemented a well-documented and reliable function that computes a metric, then move it to in `labproject/metrics` simply by add a new file `my_metric.py`.

### Plotting
If you implemented a nice plotting function, then move it to in `labproject/plotting.py`. Especially if it is applicable to multiple tasks.

### Running experiments
After committing and pushing your changes, GitHub will execute every `run_{name}.py` and update the figures in Overleaf. 

You can also run it yourself with `python labproject/run_{name}.py`. Any results, will however not be pushed into the repository.



## Documentation

We use mkdocs to create the public version of our tutorials (notebooks) and the API documentation. mkdocs are written in Markdown and are found in `docs/`.

After installing the necessary dependencies with `pip install -e ".[docs]"`, you can view your local version with `mkdocs serve` in `labproject/` and then open http://127.0.0.1:8000/.
Every push to the main branch will also publish the current version on: http://www.mackelab.org/labproject/ 

To add your notebooks as pages in the docs, add the following to `mkdocs.yml`:
```yaml
pages:
  - Notebooks: 
    - Example Notebook: notebooks/example.ipynb
    - FID notebook: notebooks/fid.ipynb
    - UR_NOTEBOOK_NAME: notebook/UR_NOTEBOOK_FILE

```

To add your functions to the API documentation, add the following to `docs/api.md`:
```markdown
### Your module
::: labproject.your_module
    options:
      heading_level: 4
```
To build it locally use `mkdocs build` (or `mkdocs serve` to view it locally). After pushing to the main branch, the documentation will be **automatically** updated.

For adding new pages, create a new markdown file in `docs/` and add it to `mkdocs.yml`:
```yaml
pages:
  - 'your_new_page.md'
```

Every docstring in the code will be automatically added to the **API documentation**. We will use "Google Style" docstrings, see [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for an example.



### Notebooks
The jupyter notebooks are found in `docs/notebooks/`.

For your convenience, at the beginning of the jupyter notebook, run    
```python
%load_ext autoreload
%autoreload 2
```
for automatic reloading of modules, in case you make any running changes to the labproject code.
