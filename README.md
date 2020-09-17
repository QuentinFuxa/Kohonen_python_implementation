# Kohonen_python_implementation

This is an optimized python implementation of unsupervised Kohonen maps, for signals (= 1d observations)
It also contains a matplotlib _FuncAnimation_ implementation, to visualize the convergence.



```python
from SOM_implementation import *

x_size = len(data[0])
size = 10
alpha = 0.05

ap = Animation_prototypes(size=size, dataset=data_norm_temp, alpha=alpha, nb_it=50000, x_size=x_size)
ap.run_animation(speed = 50)
```
