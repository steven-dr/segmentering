# Segmentering


```python
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```


```python
data = pd.read_csv("data.csv")
```

```python
replace_dict = {"Fuldstændig uenig": 1,
                "Uenig": 2,
                "Nærmest uenig": 3,
                "Nærmest enig": 4,
                "Enig": 5,
                "Fuldstændig enig": 6}

data.iloc[:,:] = data.iloc[:,:].replace(replace_dict)
```


```python
with open("scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

data_scaled = scaler.transform(data)
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
