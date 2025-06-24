#########################################################
QuickStart
#########################################################

==========================
Create your first Origami
==========================

In a jupyter file or a python file you start by importing the library:

.. code-block:: python

   from origami_tools.Patron import Patron
   from origami_tools.Geometry import * 

Create your first origami by instantiating a `Patron` object:

.. code-block:: python

   my_origami = Patron()

You can then add a shape to do the exterior of the origami:

.. code-block:: python

   my_origami.add_shapes(Rectangle.from_corner_size(Point(0, 0), 100, 100))

You can add a mountain and valley fold to the origami:

.. code-block:: python

   my_origami.add_folds(Line(Point(0, 50), Point(100, 50)), fold_type='m')
   my_origami.add_folds(Line(Point(50, 0), Point(50, 100)), fold_type='v')


You can then visualize the origami using the `show` method (work only in jupyter notebook for now):

.. code-block:: python

   my_origami.show()

To start the drawing of the origami with some margins, you can change its origin:

.. code-block:: python

   my_origami.set_origin(Point(5, 5))

You can also save the origami to a file in SVG format:

.. code-block:: python

   my_origami.save_svg("~/quickstart/origami1/", 'my_origami')

============================
Add laser cutting parameters
============================