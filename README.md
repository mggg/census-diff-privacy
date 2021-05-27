# Census TopDown: The Impacts of Differential Privacy on Redistricting

This repository contains the code and notebooks used in the preparation of the above titled paper, available at https://mggg.org/dp .
This is joint work from Aloni Cohen, Moon Duchin, JN Matthews and Bhushan Suwal.  

We thank Denis Kazakov, Mark Hansen, and Peter Wayner. Kazakov developed the reconstruction algorithm as a member of Hansen’s research group. 
Wayner guided our deployment of TopDown in AWS and was an invaluable team member for the technical report. 

## Files
The single-attribute ToyDown files and notebooks are in the ``toymodel`` directory.

The multi-attibute ToyDown files are in the  ``multi_attribute_toydown`` directory. 

All the TopDown code, as well as the pre-processing and post-processing code for the TopDown generated data is in ``topdown``.

The shapefiles and data required for the experiments on the five small Texas counties (Bell, Brazoria, Cameron, Galveston, Nueces) and Galveston City are in the ``small_counties`` directory.

The shapefiles and data required for the experiments on Dallas county is in the ``data`` directory. 

The rest of the files and files and folders are kept for various miscellaneous experiments that do not appear in the paper but are kept here for posterity.
