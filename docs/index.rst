.. hyperseti documentation master file, created by
   sphinx-quickstart on Fri Mar 24 14:41:40 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hyperseti: a narrowband radio technosignature search code
=========================================================

.. image:: figs/hyperseti-robot-web.jpg
   :width: 90 %
   :alt: hyperseti robot searching for signals
   :align: center


`Hyperseti` is a GPU-accelerated code for searching radio astronomy spectral datasets for 
narrowband technosignatures that indicate the presence of intelligent (i.e. technologically capable)
life beyond Earth. It was developed as part of the Breakthrough Listen initiative, which seeks to 
quantify the prevalence of intelligent life within the Universe.

.. toctree::
   :maxdepth: 2
   :caption: Hyperseti usage

   usage/quickstart
   usage/config
   usage/data_array
   usage/hit_db

.. toctree::
   :maxdepth: 2
   :caption: Theory of operation

   pipeline/preprocessing
   pipeline/dedoppler
   pipeline/hitsearch

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
