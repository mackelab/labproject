from invoke import task
from pathlib import Path

basepath = ""

open_cmd = "open"

fig_names = {"1": "fig"}


@task
def ConvertFile(c, fig):
    _convertsvg2pdf(c, fig)
    _convertpdf2png(c, fig)


@task
def _convertsvg2pdf(c, fig):
    pathlist = Path(f"{basepath}/{fig_names[fig]}/").glob("*.svg")
    for path in pathlist:
        c.run(f"inkscape {str(path)} --export-pdf={str(path)[:-4]}.pdf")


@task
def _convertpdf2png(c, fig):
    pathlist = Path(f"{basepath}/{fig_names[fig]}/").glob("*.pdf")
    for path in pathlist:
        c.run(f'inkscape {str(path)} --export-png={str(path)[:-4]}.png -b "white" --export-dpi=300')
