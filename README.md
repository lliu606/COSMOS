# COSMOS

This is the official (preliminary) implementation of the **COSMOS** optimizer. The implementation assumes that, for every weight-matrix tensor, `size(1)` corresponds to the input dimension, consistent with `nn.Linear`. To use the COSMOS optimizer, the code is as follows:

```
from COSMOS_optim import COSMOS

optimizer = COSMOS(model.parameters(),lr = 2e-3, betas=(0.95, 0.95), nestrov=True, weight_decay=0.1, lr_ratio=0.2)
```

Here lr_ratio is the ratio of learning rate for hidden layers to that for embedding layer.

We will release an improved version of the optimizer with lower memory usage and overhead. 
