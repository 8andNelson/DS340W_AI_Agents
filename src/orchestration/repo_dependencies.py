"""
What a cloned repository declares and what it needs installed.

Used by Master Agent's repository review (repo_review: what the repository
declares) and by the Replication Agent (what to install so the authors'
code can run).

* parse_requirements(path): each line of a requirements file, with its kind
  -- "pypi" (a package from PyPI), "url" (a VCS/URL/local install) or
  "option" (-r, -i, --extra-index-url, ...).
* imported_packages(root): the packages the repository's own code imports
  (.py files and notebook code cells), minus the standard library and the
  repository's own modules, mapped to PyPI names by IMPORT_TO_PACKAGE.
  An import that is not in that map is never installed: a guessed PyPI name
  can belong to a different package, or to a malicious one that took the
  name ("dependency confusion").
* missing_module(stderr): the module a ModuleNotFoundError names.
"""
import ast
import json
import re
import sys
import tomllib
from pathlib import Path

# Import name -> PyPI package, for the libraries research code commonly
# uses. Only these are ever installed from an import. Services that send
# data off the machine (e.g. wandb) and readers of secret files (dotenv)
# are left out on purpose.
IMPORT_TO_PACKAGE = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
    "matplotlib": "matplotlib", "seaborn": "seaborn", "plotly": "plotly", "statsmodels": "statsmodels",
    "xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost", "imblearn": "imbalanced-learn",
    "torch": "torch", "torchvision": "torchvision", "torchaudio": "torchaudio", "tensorflow": "tensorflow",
    "keras": "keras", "transformers": "transformers", "datasets": "datasets", "tokenizers": "tokenizers",
    "sentencepiece": "sentencepiece", "accelerate": "accelerate", "evaluate": "evaluate", "timm": "timm",
    "einops": "einops", "lightning": "lightning", "pytorch_lightning": "pytorch-lightning",
    "tqdm": "tqdm", "joblib": "joblib", "yaml": "pyyaml", "PIL": "pillow", "cv2": "opencv-python",
    "skimage": "scikit-image", "bs4": "beautifulsoup4", "requests": "requests", "networkx": "networkx",
    "shap": "shap", "lime": "lime", "optuna": "optuna", "hyperopt": "hyperopt", "pyswarms": "pyswarms",
    "deap": "deap", "mlxtend": "mlxtend", "nltk": "nltk", "gensim": "gensim", "spacy": "spacy",
    "sympy": "sympy", "h5py": "h5py", "tables": "tables", "openpyxl": "openpyxl", "xlrd": "xlrd",
    "xlsxwriter": "xlsxwriter", "pyarrow": "pyarrow", "polars": "polars", "numba": "numba",
    "numexpr": "numexpr", "bottleneck": "bottleneck", "dask": "dask", "tabulate": "tabulate",
    "IPython": "ipython", "nbformat": "nbformat", "nbconvert": "nbconvert", "ipykernel": "ipykernel",
    "category_encoders": "category-encoders", "umap": "umap-learn", "hdbscan": "hdbscan",
    "prophet": "prophet", "pmdarima": "pmdarima", "pyod": "pyod", "sktime": "sktime",
    "scikitplot": "scikit-plot", "graphviz": "graphviz", "pydot": "pydot", "pywt": "PyWavelets",
    "missingno": "missingno", "toml": "toml", "tomli": "tomli", "click": "click", "pydantic": "pydantic",
    "rich": "rich", "termcolor": "termcolor", "colorama": "colorama", "dill": "dill",
    "jupyter": "jupyter", "notebook": "notebook",
}
KNOWN_PACKAGES = set(IMPORT_TO_PACKAGE.values())

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$")
_URL_MARKERS = ("git+", "hg+", "svn+", "bzr+", "http://", "https://", "file:", "ftp://")
_MISSING_MODULE = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'")
_IMPORT_LINE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", re.MULTILINE)
SKIP_DIRS = {".git", "venv", ".venv", "env", "__pycache__", "node_modules", "site-packages"}


def normalize(name: str) -> str:
    """PEP 503 package name: "Scikit_Learn" -> "scikit-learn"."""
    return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()


def parse_requirement_line(raw: str) -> dict | None:
    """One requirements line -> {"line", "kind", "name", "spec"}; None for
    a blank line or comment."""
    line = raw.split(" #", 1)[0].strip()
    if not line or line.startswith("#"):
        return None
    lowered = line.lower()
    if line.startswith("-") and not lowered.startswith(("-e ", "--editable")):
        return {"line": line, "kind": "option", "name": "", "spec": ""}
    if lowered.startswith(("-e ", "--editable")) or any(m in lowered for m in _URL_MARKERS) \
            or line.startswith((".", "/")) or " @ " in line:
        return {"line": line, "kind": "url", "name": "", "spec": ""}
    match = _REQUIREMENT_NAME.match(line)
    if not match:
        return {"line": line, "kind": "url", "name": "", "spec": ""}
    return {"line": line, "kind": "pypi", "name": normalize(match.group(1)),
            "spec": match.group(3).split(";", 1)[0].strip()}


def parse_requirements(path: Path) -> list:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return [r for r in (parse_requirement_line(line) for line in text.splitlines()) if r]


def pyproject_requirements(path: Path) -> list:
    """[project].dependencies of a pyproject.toml, parsed like requirements lines."""
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    deps = (data.get("project") or {}).get("dependencies") or []
    return [r for r in (parse_requirement_line(str(d)) for d in deps) if r]


def declared_requirements(root: Path) -> list:
    """Every requirement the repository declares at its root."""
    root = Path(root)
    found = []
    if (root / "requirements.txt").is_file():
        found += parse_requirements(root / "requirements.txt")
    if (root / "pyproject.toml").is_file():
        found += pyproject_requirements(root / "pyproject.toml")
    return found


def pinned_version(spec: str) -> str:
    """"==2.2.3" -> "2.2.3"; "" when the version is not pinned exactly."""
    match = re.fullmatch(r"==\s*([A-Za-z0-9._+!-]+)", (spec or "").strip())
    return match.group(1) if match else ""


def notebook_code(path: Path) -> str:
    """The code cells of a notebook as one Python source. Shell lines (!cmd)
    and magics (%cmd) become `pass` at the same indent, so the rest parses."""
    try:
        cells = json.loads(Path(path).read_text(encoding="utf-8", errors="ignore")).get("cells") or []
    except (OSError, ValueError, AttributeError):
        return ""
    lines = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source") or ""
        source = "".join(source) if isinstance(source, list) else str(source)
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                line = line[:len(line) - len(stripped)] + "pass"
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


def notebook_shell_lines(path: Path) -> list:
    """The !shell and %magic lines of a notebook's code cells."""
    try:
        cells = json.loads(Path(path).read_text(encoding="utf-8", errors="ignore")).get("cells") or []
    except (OSError, ValueError, AttributeError):
        return []
    found = []
    for cell in cells:
        if isinstance(cell, dict) and cell.get("cell_type") == "code":
            source = cell.get("source") or ""
            source = "".join(source) if isinstance(source, list) else str(source)
            found += [line.strip() for line in source.splitlines() if line.lstrip().startswith(("!", "%"))]
    return found


def code_files(root: Path) -> list:
    """The .py and .ipynb files of a repository, outside SKIP_DIRS."""
    root = Path(root)
    return sorted(p for p in root.rglob("*")
                  if p.suffix in (".py", ".ipynb") and p.is_file()
                  and not any(part in SKIP_DIRS for part in p.relative_to(root).parts))


def _top_level_imports(source: str) -> set:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set(_IMPORT_LINE.findall(source))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def imported_packages(root: Path) -> tuple:
    """(packages, unknown): the PyPI packages the repository's code imports,
    and the third-party imports with no known package (never installed)."""
    root = Path(root)
    files = code_files(root)
    local = {p.stem for p in files} | {p.name for p in root.rglob("*") if p.is_dir()}
    imports = set()
    for path in files:
        source = notebook_code(path) if path.suffix == ".ipynb" \
            else path.read_text(encoding="utf-8", errors="ignore")
        imports |= _top_level_imports(source)
    third_party = {name for name in imports
                   if name not in sys.stdlib_module_names and name not in local and not name.startswith("_")}
    packages = sorted({IMPORT_TO_PACKAGE[name] for name in third_party if name in IMPORT_TO_PACKAGE})
    unknown = sorted(name for name in third_party if name not in IMPORT_TO_PACKAGE)
    return packages, unknown


def missing_module(stderr: str) -> str:
    """The top-level module of the last ModuleNotFoundError in stderr, or ""."""
    found = _MISSING_MODULE.findall(stderr or "")
    return found[-1].split(".")[0] if found else ""
