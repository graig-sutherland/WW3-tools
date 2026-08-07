# personal notes by Graig with getting the regional mesh to work with python
 Currently everything runs automatically if a bounding box can be defined. 
 From this a jigsawpy geom structure is created to bound the mesh

 Parameters are set in config_gjs_reg.env file and then ./create_template.sh is run to create the config file

 From this run the window mask file to set the bespoke cell size function and then run the meshing software.

 Note: Currently does not work across the North Pole. Will need to change to work on a polar stereographic grid.
