cycling-signatures gallery
==========================

Worked examples that query the Python bindings to render figures from
Lorenz-attractor, Dadras-attractor and Charney-DeVore trajectories. Each
example loads the published example data (fetched from Zenodo and cached on
first use) and renders figure(s), with the generating code shown inline.

Start with :doc:`concepts` for the vocabulary shared across every example: what
a raw, dense, or detection trajectory is, what a cover generator, class, and
signature are, and the generator-basis caveat behind the gallery's bracketed
class-vector labels. From there, the Lorenz gallery is the most approachable of
the three systems (three dimensions, two cover generators); Dadras adds a fourth
dimension and richer class structure; Charney-DeVore is a six-dimensional flow
whose loop types are not known in advance, and its section reads them from the
cycles and investigates how far that reading can be trusted. The :doc:`api`
documents the underlying classes for readers who require the full method
surface.

.. toctree::
   :maxdepth: 2

   concepts
   auto_examples/lorenz/index
   auto_examples/dadras/index
   auto_examples/charney_devore/index
   api
