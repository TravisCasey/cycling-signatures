Charney-DeVore examples
=======================

The six-mode Charney-DeVore model is a severely truncated model of mid-latitude
atmospheric flow over topography, carrying one real amplitude per retained mode.
At the forcing used here it is chaotic and switches irregularly between a fast
zonal flow and slow blocked states.

Its coordinates come in three pairs. The **zonal plane** is coordinates 1 and
4, the amplitudes of the two zonal-flow modes. The **wave plane** is coordinates
2 and 3, the cosine and sine amplitudes of the leading wave, so a turn around
the wave plane is that wave's phase advancing once. Coordinates 5 and 6 are the
same pair for the second wave. The figures that draw the state space draw one or
both of the first two planes.

Unlike Lorenz and Dadras, this system's loop types are not known in advance. Its
cover carries a thousand or more **cover generators**, the `F_2` cohomology
generators of the cover's cubical complex, and nothing about the flow says which
combinations of them the motion actually wraps. These examples are about
determining that answer and then testing how much of it survives a change in how
the system was covered. See :doc:`/concepts` for the shared vocabulary.

Reading the answer off the cycles
---------------------------------

Order the classes a storage's cycles carry by how many cycles carry them, most
frequent first: that is the **frequency order**. A prefix of it is **closed**
when its classes are exactly the nonzero elements of the span they generate,
so a closed prefix of ``2 ** r - 1`` classes has rank ``r``. The **dominant
span** is the span of the closed prefix with the largest **drop**, the step
down in cycle count from the last class of the prefix to the first class
outside it. Its **share** is the fraction of the cycles carrying a nonzero
class that its classes carry; cycles of the trivial class are outside that
denominator.

At every cube side shown, the frequency order closes exactly once, at three
classes of rank 2. Three classes spanning a rank-2 space satisfy one relation:
the sum of two of them is the third. So the reading is two independent loop
types and their sum, carried by the overwhelming majority of the cycles.

What the section shows
----------------------

Four of the examples read the dominant span off one cover at cube side 0.005
and ask what its three classes are. *Class frequency and the dominant span*
shows how far they stand above the rest in frequency and why the prefix stops
at three. *Representatives of the three classes* draws the shortest cycle
carrying each one. *Window rank against window length* asks how long a window
has to be before its signature is exactly the dominant span. *Signature
indicator* draws what single windows of four lengths read across one stretch
of the trajectory.

The other two change the cover and ask whether the reading moves. *The
signature under a rigid motion of the grid* turns and shifts the cube lattice
under the same trajectory, which leaves the detection points, and so the
cycles, untouched while rebuilding the cover from scratch in a new generator
basis. *The same classes at three cube sides* covers the same trajectory more
coarsely and more finely, which changes the cycles as well. In both cases the
three classes come back.
