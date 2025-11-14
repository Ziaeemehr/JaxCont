Pitchfork Bifurcation
=====================

This example demonstrates a classic pitchfork bifurcation using the normal form:

.. math::

   \frac{dx}{dt} = rx - x^3

Theory
------

The pitchfork bifurcation is a fundamental bifurcation where:

- For :math:`r < 0`: stable equilibrium at :math:`x = 0`
- At :math:`r = 0`: pitchfork bifurcation point
- For :math:`r > 0`: unstable equilibrium at :math:`x = 0` and stable equilibria at :math:`x = \pm\sqrt{r}`

Implementation
--------------

.. literalinclude:: ../../../examples/example_01_pitchfork.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   cd examples
   python example_01_pitchfork.py

Expected Output
---------------

The continuation should produce approximately 60 points along the branch, with a bifurcation 
detected at :math:`r \\approx 0`.

The bifurcation diagram shows:

- A stable branch for :math:`r < 0`
- Bifurcation point at :math:`r = 0`
- Unstable branch and two stable branches for :math:`r > 0`

.. image:: /_static/pitchfork_bifurcation.png
   :width: 600px
   :align: center
   :alt: Pitchfork bifurcation diagram

Variations
----------

Try modifying the system:

**Supercritical vs Subcritical**

Change to subcritical pitchfork:

.. code-block:: python

   def subcritical_pitchfork(state, params):
       x = state[0]
       r = params['r']
       return jnp.array([r * x + x**3])

**Adding Higher-Order Terms**

.. code-block:: python

   def imperfect_pitchfork(state, params):
       x = state[0]
       r = params['r']
       epsilon = params.get('epsilon', 0.0)
       return jnp.array([r * x - x**3 + epsilon])

References
----------

- Kuznetsov, Y. A. (2004). *Elements of Applied Bifurcation Theory*. Springer.
- Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos*. Westview Press.
