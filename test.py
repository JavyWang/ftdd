import pandas as pd
import time

df = pd.DataFrame({'object': ['a', 'b', 'c'], 'numeric': [1, 2, 3], 'categorical': pd.Categorical(['d','e','f'])})
print(df)
a = df.loc[0, 'object']
print(a)




