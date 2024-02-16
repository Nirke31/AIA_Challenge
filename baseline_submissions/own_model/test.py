import pandas as pd
import torch
from torch import Tensor


# Example DataFrame
data = {
    'float_column': [1.1, 2.2, 3.3, 4.4],
    'cat_column1': ['A', 'B', 'A', 'C'],
    'cat_column2': ['X', 'Y', 'X', 'Z']
}
df = pd.DataFrame(data)

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x)
print(x[1])


