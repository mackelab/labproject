
## Installation

First, clone the repository with `git clone https://github.com/mackelab/labproject.git` and then install the package `labproject` with its dependencies with `pip install -e 
.` or when you also want to edit the tutorials `pip install -e ".[docs]"`. 
You may want to do this in an environment but you can also use your standard Python installation. 

## Coding

Develop code in your desired way, e.g. in local notebooks (you don't commit, the public notebooks for the website can be found in `docs/notebooks/`). 
Once you want other people to see your figure/experiment, add them to `plotting.py` and `experiments.py` and call the corresponding functions in `run.py`. 
After committing and pushing your changes, GitHub will execute `run.py` and update the figures in Overleaf. 

You can obviously also run it yourself with `python labproject/run.py`.

## Documentation

We use mkdocs to create the public version of our tutorials (notebooks) and the API documentation. mkdocs are written in Markdown and are found in `docs/`. 
After installing the necessary dependencies with `pip install -e ".[docs]"`, you can view your local version with `mkdocs serve` in `labproject/` and then open http://127.0.0.1:8000/. 
This especially helpful when editing the documentation. 

### Notebooks
The jupyter notebooks are found in `docs/notebooks/`
