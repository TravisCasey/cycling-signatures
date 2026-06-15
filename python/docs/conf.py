"""Sphinx configuration for the cycling-signatures example gallery."""

project = "cycling-signatures"
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r".*\.py",
    "ignore_pattern": r"(_support\.py|/data/)",
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
