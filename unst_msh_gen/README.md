# some notes for gjs versions

# Global

## global landmask
The program `create_gshhs_nc.py` creates a mask based on GSHHS input file using `pygmt` to rasterize the shapefiles. 
PyGMT is available at (https://www.pygmt.org/dev/index.html).

## spacing
Currently I am using a tanh spacing function to go from 4km on the coast to 30 km off the shelf
![tanh spacing](tanhspacing.png)

$$
s = (h_{max}-h_{min}) * \frac{1}{2}\left(1 + \tanh{h_\beta\left(H-h_0\right)}\right) + h_{min}
$$
I also add a scaling to ensure that $h_{min}$ is reached for depth $H=1$ m.

