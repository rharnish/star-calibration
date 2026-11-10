"""Export article.ipynb to the article: prose and outputs, no code, no "notes" cells.

    ../../.venv/bin/python build.py            # export as it stands (prose edits)
    ../../.venv/bin/python build.py --execute  # re-run every cell first (figures, numbers)

Writes build/article.html (to read) and build/article.docx (upload to Google Drive and open
with Google Docs). The images embed in both.

--execute needs the HPWREN cache ($HPWREN_CACHE) with the article's nights and the v0.4.0
solves, and takes a while: each figure tracks and solves its nights. The two fire figures come
from plume-triangulation (plume_figs.py); where their images aren't in static/, the notebook's
saved outputs are kept.
"""
import re
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter, MarkdownExporter
from nbconvert.preprocessors import TagRemovePreprocessor
from traitlets.config import Config

HERE = Path(__file__).resolve().parent
NB = HERE / "article.ipynb"
OUT = HERE / "build"
DOCX_WIDTH = "6.5in"   # a Letter page with 1-inch margins

nb = nbformat.read(NB, as_version=4)
if "--execute" in sys.argv:
    saved = {i: c.outputs for i, c in enumerate(nb.cells) if c.cell_type == "code"}
    NotebookClient(nb, timeout=3600, resources={"metadata": {"path": str(HERE)}}).execute()
    for i, outputs in saved.items():   # F.static returned None: keep the figure we had
        if not any(o.output_type == "execute_result" for o in nb.cells[i].outputs) \
                and nb.cells[i].source.startswith("F.static("):
            nb.cells[i].outputs = outputs
    nbformat.write(nb, NB)

for cell in nb.cells:   # progress printed by the solves is for us, not the article
    if cell.cell_type == "code":
        cell.outputs = [o for o in cell.outputs if o.output_type != "stream"]

c = Config()
c.TagRemovePreprocessor.enabled = True
c.TagRemovePreprocessor.remove_cell_tags = {"notes"}
c.TemplateExporter.exclude_input = True
c.TemplateExporter.exclude_input_prompt = True
c.TemplateExporter.exclude_output_prompt = True
OUT.mkdir(exist_ok=True)

html, _ = HTMLExporter(config=c).from_notebook_node(nb)
(OUT / "article.html").write_text(html)

md, res = MarkdownExporter(config=c).from_notebook_node(nb, resources={"output_files_dir": "media"})
for old in (OUT / "media").glob("*"):
    old.unlink()
for name, data in res["outputs"].items():
    (OUT / name).parent.mkdir(exist_ok=True)
    (OUT / name).write_bytes(data)
md = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", rf"![](\1){{width={DOCX_WIDTH}}}", md)
(OUT / "article.md").write_text(md)
subprocess.run(["pandoc", "article.md", "-f", "markdown-implicit_figures", "-o", "article.docx"],
               cwd=OUT, check=True)
print("wrote", OUT / "article.html", "and", OUT / "article.docx")
