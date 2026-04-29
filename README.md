# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
## Data Cleaning

#### DEVELPOED BY : ANISE KINSELLA A
#### REG NO : 212225040021
```
import pandas as pd
data=pd.read_csv("SAMPLEIDS.csv")
data
```

<img width="965" height="732" alt="image" src="https://github.com/user-attachments/assets/55f6dc07-fe8d-46a6-88e5-b77c134cd238" />


```
data.head()

```

<img width="912" height="212" alt="image" src="https://github.com/user-attachments/assets/1049582c-6737-407d-a7ef-371ef03ec199" />


```
data.tail()
```

<img width="927" height="205" alt="image" src="https://github.com/user-attachments/assets/7a45bc40-d48c-47c8-9ac3-599dfd847284" />


```
data.isnull()
```

<img width="766" height="707" alt="image" src="https://github.com/user-attachments/assets/07ab1de5-143f-49cb-b24b-819772304449" />


```
data.isnull().sum()
```

<img width="147" height="280" alt="image" src="https://github.com/user-attachments/assets/c87d45be-1637-4b5b-b154-fac9524c99c6" />


```
data.isnull().any()
```

<img width="187" height="286" alt="image" src="https://github.com/user-attachments/assets/a55a09af-cfea-493b-b9cf-0a33232baec8" />


```
data.dropna()
```

<img width="901" height="461" alt="image" src="https://github.com/user-attachments/assets/a2896b9d-09b4-45af-84b9-7e6db8ab5545" />


```
data.fillna(0)

```

<img width="898" height="717" alt="image" src="https://github.com/user-attachments/assets/ec9ba965-0f3e-4235-bd8d-fbf7f09add8d" />


```
data.fillna(method='ffill')
```

<img width="918" height="712" alt="image" src="https://github.com/user-attachments/assets/26fa83b1-5228-4e61-804a-083ed1e8b29e" />


```
data.fillna(method='bfill')
```

<img width="953" height="716" alt="image" src="https://github.com/user-attachments/assets/10c6e194-74a7-462b-b6f0-f48bec2c06fd" />


```
 data.fillna({'GENDER':'MALE','NAME':'SRI','ADDRESS':'POONAMALEE','M1':98,'M2':87,'M3':76,'M4':92,'TOTAL':305,'AVG':89.999999})
```

<img width="907" height="720" alt="image" src="https://github.com/user-attachments/assets/7454d617-0104-497b-8c72-7cf417281082" />


### IQR(Inter Quartile Range)

```
import pandas as pd
ir=pd.read_csv("iris.csv")
ir
```

<img width="596" height="437" alt="image" src="https://github.com/user-attachments/assets/7bb97f99-ddbf-43f5-a442-a08ab9d3a208" />


```
ir.describe()
```

<img width="492" height="297" alt="image" src="https://github.com/user-attachments/assets/11010b1c-fbd9-4d2b-b262-7520b4582928" />


```
ir.shape
```

<img width="93" height="33" alt="image" src="https://github.com/user-attachments/assets/922c9a8c-2068-447a-b26a-e641ae0fa46a" />


```
ir.info()
```

<img width="458" height="257" alt="image" src="https://github.com/user-attachments/assets/f8fc81a6-cb6c-46af-b09a-410fc12aee9b" />


```
import seaborn as sns
sns.boxplot(x='sepal_width',data=ir)
```

<img width="712" height="582" alt="image" src="https://github.com/user-attachments/assets/7b4d147d-e442-472a-83de-1930eeeef4f9" />


```
 q1=ir.sepal_width.quantile(0.25)
 q3=ir.sepal_width.quantile(0.75)
 iqr=q3-q1
 print(iqr)
```

<img width="52" height="37" alt="image" src="https://github.com/user-attachments/assets/3208aff0-84b5-49ea-887b-eba8dabe334f" />


```
 out=ir[((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
 out['sepal_width']

```

<img width="326" height="116" alt="image" src="https://github.com/user-attachments/assets/7492da33-a6ea-493b-a48a-c9094fc872ba" />


```
 nor=ir[~((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
 nor['sepal_width']
```

<img width="487" height="268" alt="image" src="https://github.com/user-attachments/assets/47527a15-fd6e-4088-92c5-982c09f1f2e1" />


```
sns.boxplot(x='sepal_width',data=nor)
```

<img width="672" height="578" alt="image" src="https://github.com/user-attachments/assets/f351271d-a00b-4a1b-b2eb-049b1adeb588" />


### Z-SCORE

```
import numpy as np
import pandas as pd
df=pd.read_csv("heights.csv")
df
```

<img width="192" height="498" alt="image" src="https://github.com/user-attachments/assets/491f75ec-a62e-40a3-ba23-b902f0f21b8b" />


```
import scipy.stats as stats
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr

```

<img width="198" height="22" alt="image" src="https://github.com/user-attachments/assets/8212146f-12f2-4bb3-94a2-d8b198b91e28" />


```
low = q1 - 1.5*iqr
print(low)
high = q3 + 1.5*iqr
print(high)
```

<img width="205" height="55" alt="image" src="https://github.com/user-attachments/assets/c2a48c3a-4273-4efb-99d0-8d939fb4fdd4" />


```
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
```

<img width="213" height="433" alt="image" src="https://github.com/user-attachments/assets/665d6daa-9775-4cf2-9e19-89834fca72cf" />


```
z = np.abs(stats.zscore(df['height']))
z
```

<img width="291" height="328" alt="image" src="https://github.com/user-attachments/assets/115d7ae9-82c5-4145-88a8-3f46d4f068eb" />


```
df1 = df[z<3]
df1
```

<img width="186" height="461" alt="image" src="https://github.com/user-attachments/assets/2f7bb023-cfe8-4ba6-be8e-6e94f975260c" />

# Result
Thus we have cleaned the data and removed the outliers by detectionusing IQR and Z-score method.
