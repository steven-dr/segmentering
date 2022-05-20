# Segmentering

Segmenteringsmodellen er baseret på 10 spørgsmål:

Q1: Vi burde gøre mere for flygtninge, der kommer til Danmark, end vi gør i dag
Q2: Den offentlige sektor er for stor
Q3: Vi betaler for meget i skat i Danmark
Q4: Vi skal have styr på vores egne problemer, før vi hjælper andre lande
Q5: Enhver er sin egen lykkes smed
Q6: Jeg køber altid økologiske eller miljøvenlige produkter, hvis jeg kan
Q7: Det er vigtigt for mig at have den nyeste teknologi på markedet
Q8: Jeg kan ikke forestille mig en hverdag uden min smartphone
Q9: Jeg kommer let til at kede mig, hvis jeg laver de samme ting
Q10: Jeg vil følge moden

```python
import pickle
import pandas as pd
from sklearn.cluster import KMeans
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
with open("kmeans.pickle", "rb") as f:
    kmeans = pickle.load(f)

data["segment"] = kmeans.predict(data)
```

```python
replace_dict_segment = {0: "Selvudviklerne",
                        1: "Individualisterne",
                        2: "Pragmatikerne",
                        3: "Idealisterne",
                        4: "Beskytterne"}

data["segment"] = data["segment"].replace(replace_dict_segment)
```
