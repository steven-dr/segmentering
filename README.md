# Segmentering

Segmenteringsmodellen er udviklet i Python. Det er derfor nemmest at hente modellen direkte fra dette repository og anvende den direkte.

## Spørgsmål
Segmenteringsmodellen er baseret på disse ti spørgsmål:

1. Vi burde gøre mere for flygtninge, der kommer til Danmark, end vi gør i dag
2. Den offentlige sektor er for stor
3. Vi betaler for meget i skat i Danmark
4. Vi skal have styr på vores egne problemer, før vi hjælper andre lande
5. Enhver er sin egen lykkes smed
6. Jeg køber altid økologiske eller miljøvenlige produkter, hvis jeg kan
7. Det er vigtigt for mig at have den nyeste teknologi på markedet
8. Jeg kan ikke forestille mig en hverdag uden min smartphone
9. Jeg kommer let til at kede mig, hvis jeg laver de samme ting
10. Jeg vil følge moden

I data skal disse navngives sådan at spørgsmål 1 hedder Q1, spørgsmål 2 hedder Q2 osv.

Svarmulighederne på spørgsmålene er:
- Fuldstændig uenig
- Uenig
- Nærmest uenig
- Nærmest enig
- Enig
- Fuldstændig enig


## Kode

```python
import pickle
import pandas as pd
from sklearn.cluster import KMeans
```


```python
data = pd.read_csv("data.csv")
head(data)
```
|                Q1 |               Q2|...|            Q9|               Q10|
|-------------------|-----------------|---|--------------|------------------|
| Fuldstændig uenig | Fuldstændig enig|...|Nærmest uenig |             Uenig|
|     Nærmest uenig |             Enig|...|Nærmest uenig |     Nærmest uenig|
|              Enig |    Nærmest uenig|...|        Uenig |             Uenig|
|      Nærmest enig |    Nærmest uenig|...|Nærmest uenig |             Uenig|
|      Nærmest enig |            Uenig|...|        Uenig | Fuldstændig uenig|


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
