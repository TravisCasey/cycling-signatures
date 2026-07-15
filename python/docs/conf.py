"""Sphinx configuration for the cycling-signatures example gallery."""

import os
import sys

# The shared gallery helper lives at the examples root; sphinx-gallery puts
# only each executing example's own directory on sys.path, so add the root
# explicitly for `import _support`.
sys.path.insert(0, os.path.abspath("../examples"))

project = "cycling-signatures"
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

# Executing the examples fetches the published example data on a cache miss.
# For an offline build, pass `-D plot_gallery=0` to sphinx-build: example pages
# are parsed and rendered, but not executed, so no data is fetched.
sphinx_gallery_conf = {
    "examples_dirs": ["../examples/lorenz", "../examples/dadras"],
    "gallery_dirs": ["auto_examples/lorenz", "auto_examples/dadras"],
    "filename_pattern": r".*\.py",
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
}

autodoc_typehints = "none"
autodoc_member_order = "bysource"
add_module_names = False
numpydoc_show_class_members = False
default_role = "literal"
html_theme = "furo"
exclude_patterns = ["_build"]
