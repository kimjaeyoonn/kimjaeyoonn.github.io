---
title: '따릉이 데이터를 활용한 데이터 분석' 
excerpt: "각 날짜의 1시간 전의 기상상황을 가지고 1시간 후의 따릉이 대여수를 예측"
categories: Data-Analysis


author_profile: true    #작성자 프로필 출력 여부

last_modified_at: 2021-07-11 T21:00:00+09:00

toc: true   #Table Of Contents 목차 

toc_sticky: true
---
> 각 날짜의 1시간 전의 기상상황을 가지고 1시간 후의 따릉이 대여수를 예측해라

## Exploratory Data Analysis (EDA)


```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor    # 랜덤 포레스트 이해를 위함.
from sklearn.ensemble import RandomForestRegressor
```


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('submission.csv')
```


```python
train.head(10)
# test.head()
# submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>hour</th>
      <th>hour_bef_temperature</th>
      <th>hour_bef_precipitation</th>
      <th>hour_bef_windspeed</th>
      <th>hour_bef_humidity</th>
      <th>hour_bef_visibility</th>
      <th>hour_bef_ozone</th>
      <th>hour_bef_pm10</th>
      <th>hour_bef_pm2.5</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>20</td>
      <td>16.3</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>89.0</td>
      <td>576.0</td>
      <td>0.027</td>
      <td>76.0</td>
      <td>33.0</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>13</td>
      <td>20.1</td>
      <td>0.0</td>
      <td>1.4</td>
      <td>48.0</td>
      <td>916.0</td>
      <td>0.042</td>
      <td>73.0</td>
      <td>40.0</td>
      <td>159.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>6</td>
      <td>13.9</td>
      <td>0.0</td>
      <td>0.7</td>
      <td>79.0</td>
      <td>1382.0</td>
      <td>0.033</td>
      <td>32.0</td>
      <td>19.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>23</td>
      <td>8.1</td>
      <td>0.0</td>
      <td>2.7</td>
      <td>54.0</td>
      <td>946.0</td>
      <td>0.040</td>
      <td>75.0</td>
      <td>64.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>18</td>
      <td>29.5</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>7.0</td>
      <td>2000.0</td>
      <td>0.057</td>
      <td>27.0</td>
      <td>11.0</td>
      <td>431.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13</td>
      <td>2</td>
      <td>13.6</td>
      <td>0.0</td>
      <td>1.7</td>
      <td>80.0</td>
      <td>1073.0</td>
      <td>0.027</td>
      <td>34.0</td>
      <td>15.0</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>14</td>
      <td>3</td>
      <td>10.6</td>
      <td>0.0</td>
      <td>1.5</td>
      <td>58.0</td>
      <td>1548.0</td>
      <td>0.038</td>
      <td>62.0</td>
      <td>33.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>16</td>
      <td>21</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>21.0</td>
      <td>1961.0</td>
      <td>0.050</td>
      <td>90.0</td>
      <td>28.0</td>
      <td>146.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>19</td>
      <td>9</td>
      <td>13.8</td>
      <td>0.0</td>
      <td>1.9</td>
      <td>64.0</td>
      <td>1344.0</td>
      <td>0.039</td>
      <td>93.0</td>
      <td>19.0</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20</td>
      <td>14</td>
      <td>17.2</td>
      <td>0.0</td>
      <td>2.1</td>
      <td>32.0</td>
      <td>1571.0</td>
      <td>0.025</td>
      <td>64.0</td>
      <td>19.0</td>
      <td>83.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
# test.info()
# 머신러닝의 모델들은 입력값에 결측값이 있으면 오류가 생길 수 있음. 결측값들을 사전에 채워줘야 함.
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   id                      1459 non-null   int64  
     1   hour                    1459 non-null   int64  
     2   hour_bef_temperature    1457 non-null   float64
     3   hour_bef_precipitation  1457 non-null   float64
     4   hour_bef_windspeed      1450 non-null   float64
     5   hour_bef_humidity       1457 non-null   float64
     6   hour_bef_visibility     1457 non-null   float64
     7   hour_bef_ozone          1383 non-null   float64
     8   hour_bef_pm10           1369 non-null   float64
     9   hour_bef_pm2.5          1342 non-null   float64
     10  count                   1459 non-null   float64
    dtypes: float64(9), int64(2)
    memory usage: 125.5 KB



```python
train.describe()
# test.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>hour</th>
      <th>hour_bef_temperature</th>
      <th>hour_bef_precipitation</th>
      <th>hour_bef_windspeed</th>
      <th>hour_bef_humidity</th>
      <th>hour_bef_visibility</th>
      <th>hour_bef_ozone</th>
      <th>hour_bef_pm10</th>
      <th>hour_bef_pm2.5</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1459.000000</td>
      <td>1459.000000</td>
      <td>1457.000000</td>
      <td>1457.000000</td>
      <td>1450.000000</td>
      <td>1457.000000</td>
      <td>1457.000000</td>
      <td>1383.000000</td>
      <td>1369.000000</td>
      <td>1342.000000</td>
      <td>1459.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1105.914325</td>
      <td>11.493489</td>
      <td>16.717433</td>
      <td>0.031572</td>
      <td>2.479034</td>
      <td>52.231297</td>
      <td>1405.216884</td>
      <td>0.039149</td>
      <td>57.168736</td>
      <td>30.327124</td>
      <td>108.563400</td>
    </tr>
    <tr>
      <th>std</th>
      <td>631.338681</td>
      <td>6.922790</td>
      <td>5.239150</td>
      <td>0.174917</td>
      <td>1.378265</td>
      <td>20.370387</td>
      <td>583.131708</td>
      <td>0.019509</td>
      <td>31.771019</td>
      <td>14.713252</td>
      <td>82.631733</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>3.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>78.000000</td>
      <td>0.003000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>555.500000</td>
      <td>5.500000</td>
      <td>12.800000</td>
      <td>0.000000</td>
      <td>1.400000</td>
      <td>36.000000</td>
      <td>879.000000</td>
      <td>0.025500</td>
      <td>36.000000</td>
      <td>20.000000</td>
      <td>37.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1115.000000</td>
      <td>11.000000</td>
      <td>16.600000</td>
      <td>0.000000</td>
      <td>2.300000</td>
      <td>51.000000</td>
      <td>1577.000000</td>
      <td>0.039000</td>
      <td>51.000000</td>
      <td>26.000000</td>
      <td>96.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1651.000000</td>
      <td>17.500000</td>
      <td>20.100000</td>
      <td>0.000000</td>
      <td>3.400000</td>
      <td>69.000000</td>
      <td>1994.000000</td>
      <td>0.052000</td>
      <td>69.000000</td>
      <td>37.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2179.000000</td>
      <td>23.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>99.000000</td>
      <td>2000.000000</td>
      <td>0.125000</td>
      <td>269.000000</td>
      <td>90.000000</td>
      <td>431.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.groupby('hour').mean()['count'].plot()    # 시간별 평균 대여량 산출
```




    <AxesSubplot:xlabel='hour'>



  ![output_7_1.png](https://github.com/kimjaeyoonn/kimjaeyoonn.github.io/blob/master/_images/Seoul%20Cycle/output_7_1.png?raw=true)



```python
import matplotlib.pyplot as plt
plt.plot(train.groupby('hour').mean()['count'], 'o-')
plt.grid()    # 보조선 추가
```

​    ![output_8_0.png](https://github.com/kimjaeyoonn/kimjaeyoonn.github.io/blob/master/_images/Seoul%20Cycle/output_8_0.png?raw=true)

```python
plt.plot(train.groupby('hour').mean()['count'], 'o-')
plt.grid()

plt.title('count by hours', fontsize = 15)
plt.xlabel('hour', fontsize = 15)
plt.ylabel('count', fontsize = 15)
```




    Text(0, 0.5, 'count')

![output_9_1.png](https://github.com/kimjaeyoonn/kimjaeyoonn.github.io/blob/master/_images/Seoul%20Cycle/output_9_1.png?raw=true)

```python
plt.plot(train.groupby('hour').mean()['count'], 'o-')
plt.grid()

plt.title('count by hours', fontsize = 15)
plt.xlabel('hour', fontsize = 15)
plt.ylabel('count', fontsize = 15)

plt.axvline(8, color = 'r')
plt.axvline(18, color = 'r')    # 빨간선 표시

plt.text(8, 120, 'go work', fontsize = 10)
plt.text(18, 120, 'leave work', fontsize = 10)    # 텍스트 표시
```




    Text(18, 120, 'leave work')

![output_10_1.png](https://github.com/kimjaeyoonn/kimjaeyoonn.github.io/blob/master/_images/Seoul%20Cycle/output_10_1.png?raw=true)

```python
import seaborn as sns    # 상관계수 한눈에 알아보기 위해 유용한 라이브러리 
```


```python
# count와 상관계수가 높은 변수들을 골라서 모델을 생성한다.
plt.figure(figsize=(10, 10))
sns.heatmap(train.corr(), annot = True)    # annot : 수치도 표시

# 3가지로 모델링 해보자 (시간, 온도, 풍속)
```




    <AxesSubplot:>

  ![output_12_1.png](https://github.com/kimjaeyoonn/kimjaeyoonn.github.io/blob/master/_images/Seoul%20Cycle/output_12_1.png?raw=true)

데이터 전처리

● 데이터 프레임을 최적의 형태로 만들어 주는 과정. 결측치에 관심을 가져야함.


```python
train.isna().sum()

# 결측치면 True, 아니면 False 반환
# 온도, 풍속 변수에 결측치 2개, 9개 존재
```




    id                          0
    hour                        0
    hour_bef_temperature        2
    hour_bef_precipitation      2
    hour_bef_windspeed          9
    hour_bef_humidity           2
    hour_bef_visibility         2
    hour_bef_ozone             76
    hour_bef_pm10              90
    hour_bef_pm2.5            117
    count                       0
    dtype: int64




```python
train[train['hour_bef_temperature'].isna()]
train[train['hour_bef_windspeed'].isna()]

# 자정과 18시에 결측치 존재
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>hour</th>
      <th>hour_bef_temperature</th>
      <th>hour_bef_precipitation</th>
      <th>hour_bef_windspeed</th>
      <th>hour_bef_humidity</th>
      <th>hour_bef_visibility</th>
      <th>hour_bef_ozone</th>
      <th>hour_bef_pm10</th>
      <th>hour_bef_pm2.5</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>33</td>
      <td>13</td>
      <td>22.6</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>41.0</td>
      <td>987.0</td>
      <td>0.046</td>
      <td>64.0</td>
      <td>39.0</td>
      <td>208.0</td>
    </tr>
    <tr>
      <th>244</th>
      <td>381</td>
      <td>1</td>
      <td>14.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>55.0</td>
      <td>1992.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>260</th>
      <td>404</td>
      <td>3</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>50.0</td>
      <td>2000.0</td>
      <td>0.049</td>
      <td>35.0</td>
      <td>22.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>376</th>
      <td>570</td>
      <td>0</td>
      <td>14.3</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>49.0</td>
      <td>2000.0</td>
      <td>0.044</td>
      <td>37.0</td>
      <td>20.0</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>780</th>
      <td>1196</td>
      <td>20</td>
      <td>16.5</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>31.0</td>
      <td>2000.0</td>
      <td>0.058</td>
      <td>39.0</td>
      <td>18.0</td>
      <td>181.0</td>
    </tr>
    <tr>
      <th>934</th>
      <td>1420</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>1035</th>
      <td>1553</td>
      <td>18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1138</th>
      <td>1717</td>
      <td>12</td>
      <td>21.4</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>1375.0</td>
      <td>0.044</td>
      <td>61.0</td>
      <td>37.0</td>
      <td>116.0</td>
    </tr>
    <tr>
      <th>1229</th>
      <td>1855</td>
      <td>2</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>52.0</td>
      <td>2000.0</td>
      <td>0.044</td>
      <td>37.0</td>
      <td>20.0</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 온도 변화

train.groupby('hour').mean()['hour_bef_temperature'].plot()
plt.axhline(train.groupby('hour').mean()['hour_bef_temperature'].mean())

# 0시의 평균온도와와 전체 평균온도로 대체한다면, 모델링 결과에 차이가 크게 나타날 것임을 알았음.
# 전체 평균온도를 넣기보다는 각 시간별 평균온도로 대체함으로써 시간대별 경향성을 반영해야겠다.
```




    <matplotlib.lines.Line2D at 0x203391f78e0>

![output_16_1.png](https://github.com/kimjaeyoonn/kimjaeyoonn.github.io/blob/master/_images/Seoul%20Cycle/output_16_1.png?raw=true)

```python
# 시간대별 평균 온도 알아봄.

train.groupby('hour').mean()['hour_bef_temperature']
train.groupby('hour').mean()['hour_bef_windspeed']
```




    hour
    0     1.965517
    1     1.836667
    2     1.633333
    3     1.620000
    4     1.409836
    5     1.296721
    6     1.331148
    7     1.262295
    8     1.632787
    9     1.829508
    10    2.122951
    11    2.485246
    12    2.766667
    13    3.281356
    14    3.522951
    15    3.768852
    16    3.820000
    17    3.801667
    18    3.838333
    19    3.595082
    20    3.278333
    21    2.755000
    22    2.498361
    23    2.195082
    Name: hour_bef_windspeed, dtype: float64




```python
# 결측치 채워줄 때, 딕셔너리 사용

train['hour_bef_temperature'].fillna({934:14.788136, 1035: 20.926667}, inplace = True)    # inplace = True 로 저장까지 완료
train['hour_bef_windspeed'].fillna({18:3.281356, 244:1.836667, 260:1.620000, 376:1.965517, 780:3.278333, 934:1.965517, 1035:3.838333, 1138:2.766667, 1229:1.633333}, inplace = True)
```


```python
train.isna().sum()    # 전체적인 결측치 확인

test[test['hour_bef_temperature'].isna()]    # 해당 칼럼의 결측치가 포함된 열 확인
test['hour_bef_temperature'].fillna(19.704918, inplace = True)    # train의 시간대별 평균온도로 채워줌

test[test['hour_bef_windspeed'].isna()]     # 해당 칼럼의 결측치가 포함된 열 확인
test['hour_bef_windspeed'].fillna(3.595082, inplace = True)    # train의 시간대별 평균풍속으로 채워줌
```

## 모델링
● train에서 시간, 온도, 풍속 변수를 뽑아 1시간 후의 따릉이 대여량 예측하는 모델


```python
features = ['hour', 'hour_bef_temperature', 'hour_bef_windspeed']
X_train = train[features]    # 모델 학습 시 사용
y_train = train['count']    # 모델 학습 시 사용
X_test = test[features]    # 모델의 성능 테스트
```


```python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
```

    (1459, 3)
    (1459,)
    (715, 3)



```python
# 랜덤 포레스트 모형 설정

model100 = RandomForestRegressor(n_estimators=100, random_state=0)
model100_5 = RandomForestRegressor(n_estimators=100, max_depth = 5, random_state=0)
model200 = RandomForestRegressor(n_estimators=200)
```


```python
# 모델 학습

model100.fit(X_train, y_train)
model100_5.fit(X_train, y_train)
model200.fit(X_train, y_train)
```




    RandomForestRegressor(n_estimators=200)




```python
# 모델 예측

ypred1 = model100.predict(X_test)
ypred2 = model100_5.predict(X_test)
ypred3 = model200.predict(X_test)
```


```python
# submission 파일에 예측한 결과 저장

submission['count'] = ypred1
submission.to_csv('model100.csv', index = False)

submission['count'] = ypred2
submission.to_csv('model100_5.csv', index = False)

submission['count'] = ypred3
submission.to_csv('model200.csv', index = False)
```
