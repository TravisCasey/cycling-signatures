cycling-signatures gallery
==========================

Worked examples that query the Python bindings to render figures from Lorenz-
and Dadras-attractor trajectories. Each example loads the published example
data (fetched from Zenodo and cached on first use) and renders figure(s), with
the generating code shown inline.

Start with :doc:`concepts` for the vocabulary shared across every example: what
a raw, dense, or detection trajectory is, what a cover generator, class, and
signature are, and the generator-basis caveat behind the gallery's bracketed
class-vector labels. From there, the Lorenz gallery is the more approachable of
the two systems (three dimensions, two cover generators); Dadras adds a fourth
dimension and richer class structure. The :doc:`api` documents the underlying
classes for readers who require the full method surface.

.. toctree::
   :maxdepth: 2

   concepts
   auto_examples/lorenz/index
   auto_examples/dadras/index
   api
