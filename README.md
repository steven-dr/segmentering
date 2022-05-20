# Segmentering


```python
import pickle
import pandas as pd
from sklearn.cluster import KMeans
```


```python
data = pd.read_csv("data.csv")
```

```python
replace_dict_segment = {0: "Selvudviklerne",
                        1: "Individualisterne",
                        2: "Pragmatikerne",
                        3: "Idealisterne",
                        4: "Beskytterne"}

data.iloc[:,:] = data.iloc[:,:].replace(replace_dict)
```


```python
with open("kmeans.pickle", "rb") as f:
    kmeans = pickle.load(f)

data["segment"] = kmeans.predict(data_scaled)
```

```python
replace_dict_segment = {0: "De Urbane",
                        1: "De Udfordrende",
                        2: "De Sammenholdende",
                        3: "De Velgørende",
                        4: "De Beskyttende"}

data["segment"] = data["segment"].replace(replace_dict_segment)
```
