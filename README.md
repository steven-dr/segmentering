# Segmentering

Segmenteringsmodellen er udviklet i Python med sklearn (version 1.0). Modellen kan hentes direkte fra dette repository. Herunder er et Python eksempel, hvor modellen anvendt på et mindre datasæt.

Dokumentation:
 - sklearn https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
 - kmeans https://en.wikipedia.org/wiki/K-means_clustering

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
1. Fuldstændig uenig
2. Uenig
3. Nærmest uenig
4. Nærmest enig
5. Enig
6. Fuldstændig enig


## Kode
Importerer de nødvendige Python biblioteker:
```python
import pickle
import pandas as pd
from sklearn.cluster import KMeans
```
### Data
Indlæser og inspicerer data:
```python
data = pd.read_csv("data.csv")
print(data.head())
```
|<sup>id| <sup>Q1</sup> | <sup>Q2</sup> | <sup>Q3</sup>  | <sup>Q4</sup> | <sup>Q5</sup>    | <sup>Q6</sup>     | <sup>Q7</sup>  | <sup>Q8</sup>   | <sup>Q9</sup>   | <sup>Q10</sup>               |
|---|-------------------|------------------|------------------|------------------|------------------|-------------------|---------------|------------------|---------------|-------------------|
|<sup>0| <sup><sup>Fuldstændig uenig</sup></sup> | <sup><sup>Fuldstændig enig</sup></sup> | <sup><sup>Uenig</sup></sup>            | <sup><sup>Fuldstændig enig</sup></sup> | <sup><sup>Fuldstændig enig</sup></sup> | <sup><sup>Nærmest enig</sup></sup>      | <sup><sup>Nærmest uenig</sup></sup> | <sup><sup>Nærmest enig</sup></sup>     | <sup><sup>Nærmest uenig</sup></sup> | <sup><sup>Uenig</sup></sup>             |
|<sup>1| <sup><sup>Nærmest uenig</sup></sup>     | <sup><sup>Enig</sup></sup>             | <sup><sup>Fuldstændig enig</sup></sup> | <sup><sup>Enig</sup></sup>              | <sup><sup>Enig</sup></sup>            | <sup><sup>Nærmest uenig</sup></sup>     | <sup><sup>Nærmest uenig</sup></sup> | <sup><sup>Fuldstændig enig</sup></sup> | <sup><sup>Nærmest uenig</sup></sup> | <sup><sup>Nærmest uenig</sup></sup>     |
|<sup>2| <sup><sup>Enig              | <sup><sup>Nærmest uenig    | <sup><sup>Enig             | <sup><sup>Nærmest enig     | <sup><sup>Enig             | <sup><sup>Fuldstændig uenig | <sup><sup>Uenig         | <sup><sup>Nærmest enig     | <sup><sup>Uenig         | <sup><sup>Uenig             |
|<sup>3| <sup><sup>Nærmest enig      | <sup><sup>Nærmest uenig    | <sup><sup>Nærmest uenig    | <sup><sup>Nærmest uenig    | <sup><sup>Nærmest enig     | <sup><sup>Fuldstændig uenig | <sup><sup>Uenig         | <sup><sup>Uenig            | <sup><sup>Nærmest uenig | <sup><sup>Uenig             |
|<sup>4| <sup><sup>Nærmest enig      | <sup><sup>Uenig            | <sup><sup>Uenig            | <sup><sup>Nærmest uenig    | <sup><sup>Nærmest uenig    | <sup><sup>Nærmest uenig     | <sup><sup>Nærmest uenig | <sup><sup>Nærmest enig     | <sup><sup>Uenig         | <sup><sup>Fuldstændig uenig |
|...| ...| ...| ...| ...| ...| ...| ...| ...| ...| ...|
 
<sup>Denne tabel indeholder de fem første observationer af 'data'</sup>

I dette datasæt er svarene i tekstformat. Kmeans-modellen kan ikke anvendes på tekstdata. Derfor skal teksten i kolonnerne ændres til numeriske værdier:

```python
replace_dict = {"Fuldstændig uenig": 1,
                "Uenig": 2,
                "Nærmest uenig": 3,
                "Nærmest enig": 4,
                "Enig": 5,
                "Fuldstændig enig": 6}

data.iloc[:,:] = data.iloc[:,:].replace(replace_dict)
print(data.head())
```
|id| Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 |
|---|----|----|----|----|----|----|----|----|----|-----|
|0| 1  | 6  | 2  | 6  | 6  | 4  | 3  | 4  | 3  | 2   |
|1| 3  | 5  | 6  | 5  | 5  | 3  | 3  | 6  | 3  | 3   |
|2| 5  | 3  | 5  | 4  | 5  | 1  | 2  | 4  | 2  | 2   |
|3| 4  | 3  | 3  | 3  | 4  | 1  | 2  | 2  | 3  | 2   |
|4| 4  | 2  | 2  | 3  | 3  | 3  | 3  | 4  | 2  | 1   |
|...| ...| ...| ...| ...| ...| ...| ...| ...| ...| ... |

<sup>Denne tabel indeholder de fem første observationer af 'data'</sup>

Data er nu numerisk.

### Model
Modellen loades og benyttes på data:
```python
with open("kmeans.pickle", "rb") as f:
    kmeans = pickle.load(f)

data["segment"] = kmeans.predict(data)
```

Segmenterne kommer i første omgang ud som numeriske værdier. Her ændres værdierne til navnene på segmenterne:
```python
replace_dict_segment = {0: "Selvudviklerne",
                        1: "Individualisterne",
                        2: "Pragmatikerne",
                        3: "Idealisterne",
                        4: "Beskytterne"}

data["segment"] = data["segment"].replace(replace_dict_segment)
print(data.head())
```

|id| Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 | segment           |
|---|----|----|----|----|----|----|----|----|----|-----|-------------------|
|0| 1  | 6  | 2  | 6  | 6  | 4  | 3  | 4  | 3  | 2   | Beskytterne       |
|1| 3  | 5  | 6  | 5  | 5  | 3  | 3  | 6  | 3  | 3   | Individualisterne |
|2| 5  | 3  | 5  | 4  | 5  | 1  | 2  | 4  | 2  | 2   | Pragmatikerne     |
|3| 4  | 3  | 3  | 3  | 4  | 1  | 2  | 2  | 3  | 2   | Pragmatikerne     |
|4| 4  | 2  | 2  | 3  | 3  | 3  | 3  | 4  | 2  | 1   | Idealisterne      |
|...| ...| ...| ...| ...| ...| ...| ...| ...| ...| ... | ...               |

<sup>Denne tabel indeholder de fem første observationer af 'data'</sup>

I ovenstående tabel er resultatet af segmenteringsmodellen på de fem første observationer tilføjet i kolonnen 'segment'

