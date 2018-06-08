import numpy as np
import pandas as pd
import seaborn as sns

df = sns.load_dataset('iris')
sns.pairplot(df)
