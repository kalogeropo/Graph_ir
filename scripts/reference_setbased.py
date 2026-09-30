#!/usr/bin/env python3
"""
Reference run of old SetBased model on baeza, test, and CF collections.

This script:
1. Sets up an isolated venv with legacy library versions
2. Copies collection JSONL data to a format the old code expects
3. Runs SetBased and saves rankings/scores as fixtures

Run from the repository root:
  python scripts/reference_setbased.py

Outputs are saved to tests/fixtures/setbased_reference/ with metadata.
The old code stays under to_merge/ and is not committed.
"""

import sys
import subprocess
import tempfile
import shutil
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TO_MERGE = ROOT / "to_merge" / "Gsb_model"
COLLECTIONS = ROOT / "collections"
FIXTURES = ROOT / "tests" / "fixtures" / "setbased_reference"

LEGACY_REQS = [
    "nltk==3.6.5",
    "numpy==1.23.4",
    "networkx==2.6.3",
    "pandas==1.5.2",
    "scikit-learn==1.1.3",
]


def run(cmd, cwd=None, check=True):
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=check
    )
    if result.returncode != 0 and check:
        print(f"Command failed: {cmd}", file=sys.stderr)
        print(f"stdout: {result.stdout}", file=sys.stderr)
        print(f"stderr: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main():
    print("Reference SetBased run: stage 0 baseline")
    print(f"Using source: {TO_MERGE}")
    print(f"Output: {FIXTURES}")

    if not TO_MERGE.exists():
        print(f"Error: {TO_MERGE} not found", file=sys.stderr)
        sys.exit(1)

    FIXTURES.mkdir(parents=True, exist_ok=True)

    # Create a throwaway venv with legacy libraries
    with tempfile.TemporaryDirectory(prefix="graph_ir_ref_") as tmpdir:
        venv_path = Path(tmpdir) / "venv"
        print(f"\nSetting up reference venv at {venv_path}...")
        run(f"python3.12 -m venv {venv_path}")

        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"

        # Install legacy dependencies
        print("Installing legacy libraries...")
        for req in LEGACY_REQS:
            run(f"{pip_exe} install -q {req}")

        # Convert new collections to the old code's format (docs directory)
        for collection_name in ["baeza", "test", "cf"]:
            collection_dir = COLLECTIONS / collection_name
            if not collection_dir.exists():
                print(f"Skipping {collection_name}: not found")
                continue

            work_dir = Path(tmpdir) / collection_name
            work_dir.mkdir()
            docs_dir = work_dir / "docs"
            docs_dir.mkdir()

            # Load JSONL corpus and write as old-style text files
            corpus_path = collection_dir / "corpus.jsonl"
            with open(corpus_path) as f:
                for line in f:
                    record = json.loads(line)
                    doc_id = record["_id"]
                    text = "\n".join(
                        record.get(field, "")
                        for field in ["title", "text", "abstract", "extract"]
                        if record.get(field)
                    ).strip()
                    doc_file = docs_dir / str(doc_id)
                    doc_file.write_text(text)

            # Write queries and qrels
            queries_path = collection_dir / "queries.jsonl"
            qrels_path = collection_dir / "qrels.tsv"

            queries = []
            with open(queries_path) as f:
                for line in f:
                    record = json.loads(line)
                    queries.append(record["text"])

            (work_dir / "Queries.txt").write_text("\n".join(queries))

            # Parse qrels to old format (query_id: [doc_ids...])
            relevant = {}
            with open(qrels_path) as f:
                header = f.readline()  # skip
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        qid, docid, score = parts[0], parts[1], int(parts[2])
                        if score >= 1:
                            if qid not in relevant:
                                relevant[qid] = []
                            relevant[qid].append(int(docid))

            # Run reference SetBased
            print(f"\nRunning SetBased on {collection_name}...")
            run_script = f"""
import sys
sys.path.insert(0, "{TO_MERGE}")
from Preprocess.Collection import Collection
from models.SetBased import SetBasedModel
import numpy as np
import json

# Monkey-patch SetBased._model_func to return ones (no weighting)
def fixed_model_func(self, freq_termsets):
    return np.ones(len(freq_termsets))

SetBasedModel._model_func = fixed_model_func

# Load collection
col = Collection("{work_dir}/docs", name="{collection_name}")
col.create_collection()

queries_text = open("{work_dir}/Queries.txt").read().split("\\n")
queries = [q.strip().split() for q in queries_text if q.strip()]
col.queries = queries

relevant = {json.dumps(relevant)}
col.relevant = {[relevant.get(str(i+1), []) for i in range(len(queries))]}

# Run model
model = SetBasedModel(col)
model.fit(min_freq=1)
precisions, recalls = model.evaluate(k=None)

# Extract rankings
output = {{
    "collection": "{collection_name}",
    "queries": queries,
    "rankings": [list(r) for r in model.ranking],
    "precisions": [float(p) for p in precisions],
    "recalls": [float(r) for r in recalls],
}}
print(json.dumps(output, indent=2))
"""
            output = run(f'{python_exe} -c "{run_script}"')
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                print(
                    f"Failed to parse output from {collection_name}",
                    file=sys.stderr,
                )
                print(f"Output was: {output}", file=sys.stderr)
                continue

            # Save fixture
            fixture_file = FIXTURES / f"{collection_name}.json"
            with open(fixture_file, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "source": "to_merge/Gsb_model",
                            "model": "SetBased",
                            "fix": "SetBased._model_func = lambda self, ts: np.ones(len(ts))",
                            "libraries": LEGACY_REQS,
                        },
                        "data": data,
                    },
                    f,
                    indent=2,
                )
            print(f"Saved: {fixture_file}")

    print("\nReference stage complete. Fixtures saved in tests/fixtures/setbased_reference/")


if __name__ == "__main__":
    main()
