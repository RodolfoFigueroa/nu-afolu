import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")

with app.setup:
    import ee
    ee.Initialize()

    import marimo as mo

    import os
    from pathlib import Path
    from nu_afolu.chen import load_chen_analysis_zones


@app.cell
def _():
    out_path = Path(os.environ["OUT_PATH"])
    return (out_path,)


@app.cell
def _(out_path):
    col = load_chen_analysis_zones(out_path, ["01.1.01"])
    return (col,)


@app.cell
def _(col):
    col["01.1.01"].transition_arr
    return


if __name__ == "__main__":
    app.run()
