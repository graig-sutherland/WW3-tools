# Personal notes by Graig with getting the regional mesh to work with python

 Currently everything runs automatically if a bounding box can be defined. 
 From this a jigsawpy geom structure is created to bound the mesh.
 Then I create a custom spacing domain and run the meshing software.

 1. Parameters are set in `config_gjs_reg.env` file and then `./create_template.sh` is run to create the config file and save to `config_gjs_reg.ini`
 2. Create geometry file for subgrid `python create_geom.py --config config_gjs_reg.ini`
 3. From this run the window mask file to set the bespoke cell size function and then run the meshing software.
 4. Run `python ocn_ww3_gshhs_regional.py --config config_gjs_reg.ini` to run meshing software. This will be saved in the `./data` directory
