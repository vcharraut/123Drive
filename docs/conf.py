import tomllib
from pathlib import Path


project = "123Drive"
author = "Valentin Charraut"
copyright = "2026, Valentin Charraut"

metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
release = metadata["project"]["version"]
version = release

extensions = []
source_suffix = {".rst": "restructuredtext"}
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "123Drive"
html_baseurl = "https://vcharraut.github.io/123Drive/"
html_theme_options = {
    "source_repository": "https://github.com/vcharraut/123Drive/",
    "source_branch": "main",
    "source_directory": "docs/",
}
