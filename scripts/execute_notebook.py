"""Offline code-cell validator/executor; not a replacement for Jupyter kernel QA.

Skips network/setup/optional-Colab cells, compiles every code cell, executes the
remaining cells in one Python namespace, and saves captured stdout as notebook
outputs. Run only on the trusted generated notebook00.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import time

path=Path(sys.argv[1])
notebook=json.loads(path.read_text())
assert notebook["nbformat"]==4 and notebook["nbformat_minor"]==5
namespace={"__name__":"__main__"}
count=0
for cell in notebook["cells"]:
    if cell["cell_type"]!="code":continue
    source=cell["source"]
    if isinstance(source,list):source="".join(source)
    compiled=compile(source,f"{path.name}:{cell['id']}","exec")
    if set(cell["metadata"].get("tags",[])) & {"requires-network","setup","optional-colab"}:
        continue
    count+=1
    start=time.perf_counter()
    output=io.StringIO()
    with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
        exec(compiled,namespace)
    cell["execution_count"]=count
    cell["outputs"]=[{"output_type":"stream","name":"stdout","text":output.getvalue()}]
    print(cell["id"],f"{time.perf_counter()-start:.2f}s")
destination=Path(sys.argv[2]) if len(sys.argv)>2 else path.with_name(path.stem+".executed.ipynb")
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(notebook,indent=2)+"\n")
print("Saved executed code cells to",destination)
