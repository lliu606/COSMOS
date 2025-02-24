# COSMOS

This is the official (preliminary) implementation of the COSMOS optimizer. To use, copy the COSMOS.py file to your codebase and use COSMOS optimizer in the following fashion:

```
from COSMOS import COSMOS

optimizer = COSMOS(lr = 1e-3, betas=(0.9, 0.95, 0.95), lr_ratio=0.1)
```
Here lr_ratio is the ratio of learning rate for hidden layers to that for embedding layer.

We will release an improved version of the optimizer with lower memory usage and overhead. 
