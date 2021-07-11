---
title: "Jekyll Github 블로그에 MathJax 적용하기"
date: 2020-03-08 12:14:00 +0800
categories: [Blogging, Configuration]
tags: [Jekyll, MathJax]
use_math: true
---

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 따릉이 데이터를 활용한 데이터 분석\n",
    "● 각 날짜의 1시간 전의 기상상황을 가지고 1시간 후의 따릉이 대여수를 예측해라"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Exploratory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.tree import DecisionTreeRegressor    # 랜덤 포레스트 이해를 위함.\n",
    "from sklearn.ensemble import RandomForestRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "train = pd.read_csv('train.csv')\n",
    "test = pd.read_csv('test.csv')\n",
    "submission = pd.read_csv('submission.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>hour</th>\n",
       "      <th>hour_bef_temperature</th>\n",
       "      <th>hour_bef_precipitation</th>\n",
       "      <th>hour_bef_windspeed</th>\n",
       "      <th>hour_bef_humidity</th>\n",
       "      <th>hour_bef_visibility</th>\n",
       "      <th>hour_bef_ozone</th>\n",
       "      <th>hour_bef_pm10</th>\n",
       "      <th>hour_bef_pm2.5</th>\n",
       "      <th>count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>20</td>\n",
       "      <td>16.3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.5</td>\n",
       "      <td>89.0</td>\n",
       "      <td>576.0</td>\n",
       "      <td>0.027</td>\n",
       "      <td>76.0</td>\n",
       "      <td>33.0</td>\n",
       "      <td>49.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>6</td>\n",
       "      <td>13</td>\n",
       "      <td>20.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>48.0</td>\n",
       "      <td>916.0</td>\n",
       "      <td>0.042</td>\n",
       "      <td>73.0</td>\n",
       "      <td>40.0</td>\n",
       "      <td>159.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>7</td>\n",
       "      <td>6</td>\n",
       "      <td>13.9</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.7</td>\n",
       "      <td>79.0</td>\n",
       "      <td>1382.0</td>\n",
       "      <td>0.033</td>\n",
       "      <td>32.0</td>\n",
       "      <td>19.0</td>\n",
       "      <td>26.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8</td>\n",
       "      <td>23</td>\n",
       "      <td>8.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.7</td>\n",
       "      <td>54.0</td>\n",
       "      <td>946.0</td>\n",
       "      <td>0.040</td>\n",
       "      <td>75.0</td>\n",
       "      <td>64.0</td>\n",
       "      <td>57.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9</td>\n",
       "      <td>18</td>\n",
       "      <td>29.5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4.8</td>\n",
       "      <td>7.0</td>\n",
       "      <td>2000.0</td>\n",
       "      <td>0.057</td>\n",
       "      <td>27.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>431.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>13</td>\n",
       "      <td>2</td>\n",
       "      <td>13.6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.7</td>\n",
       "      <td>80.0</td>\n",
       "      <td>1073.0</td>\n",
       "      <td>0.027</td>\n",
       "      <td>34.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>39.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>14</td>\n",
       "      <td>3</td>\n",
       "      <td>10.6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.5</td>\n",
       "      <td>58.0</td>\n",
       "      <td>1548.0</td>\n",
       "      <td>0.038</td>\n",
       "      <td>62.0</td>\n",
       "      <td>33.0</td>\n",
       "      <td>23.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>16</td>\n",
       "      <td>21</td>\n",
       "      <td>16.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>21.0</td>\n",
       "      <td>1961.0</td>\n",
       "      <td>0.050</td>\n",
       "      <td>90.0</td>\n",
       "      <td>28.0</td>\n",
       "      <td>146.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>19</td>\n",
       "      <td>9</td>\n",
       "      <td>13.8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.9</td>\n",
       "      <td>64.0</td>\n",
       "      <td>1344.0</td>\n",
       "      <td>0.039</td>\n",
       "      <td>93.0</td>\n",
       "      <td>19.0</td>\n",
       "      <td>39.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>20</td>\n",
       "      <td>14</td>\n",
       "      <td>17.2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.1</td>\n",
       "      <td>32.0</td>\n",
       "      <td>1571.0</td>\n",
       "      <td>0.025</td>\n",
       "      <td>64.0</td>\n",
       "      <td>19.0</td>\n",
       "      <td>83.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  \\\n",
       "0   3    20                  16.3                     1.0                 1.5   \n",
       "1   6    13                  20.1                     0.0                 1.4   \n",
       "2   7     6                  13.9                     0.0                 0.7   \n",
       "3   8    23                   8.1                     0.0                 2.7   \n",
       "4   9    18                  29.5                     0.0                 4.8   \n",
       "5  13     2                  13.6                     0.0                 1.7   \n",
       "6  14     3                  10.6                     0.0                 1.5   \n",
       "7  16    21                  16.0                     0.0                 6.0   \n",
       "8  19     9                  13.8                     0.0                 1.9   \n",
       "9  20    14                  17.2                     0.0                 2.1   \n",
       "\n",
       "   hour_bef_humidity  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  \\\n",
       "0               89.0                576.0           0.027           76.0   \n",
       "1               48.0                916.0           0.042           73.0   \n",
       "2               79.0               1382.0           0.033           32.0   \n",
       "3               54.0                946.0           0.040           75.0   \n",
       "4                7.0               2000.0           0.057           27.0   \n",
       "5               80.0               1073.0           0.027           34.0   \n",
       "6               58.0               1548.0           0.038           62.0   \n",
       "7               21.0               1961.0           0.050           90.0   \n",
       "8               64.0               1344.0           0.039           93.0   \n",
       "9               32.0               1571.0           0.025           64.0   \n",
       "\n",
       "   hour_bef_pm2.5  count  \n",
       "0            33.0   49.0  \n",
       "1            40.0  159.0  \n",
       "2            19.0   26.0  \n",
       "3            64.0   57.0  \n",
       "4            11.0  431.0  \n",
       "5            15.0   39.0  \n",
       "6            33.0   23.0  \n",
       "7            28.0  146.0  \n",
       "8            19.0   39.0  \n",
       "9            19.0   83.0  "
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head(10)\n",
    "# test.head()\n",
    "# submission.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1459 entries, 0 to 1458\n",
      "Data columns (total 11 columns):\n",
      " #   Column                  Non-Null Count  Dtype  \n",
      "---  ------                  --------------  -----  \n",
      " 0   id                      1459 non-null   int64  \n",
      " 1   hour                    1459 non-null   int64  \n",
      " 2   hour_bef_temperature    1457 non-null   float64\n",
      " 3   hour_bef_precipitation  1457 non-null   float64\n",
      " 4   hour_bef_windspeed      1450 non-null   float64\n",
      " 5   hour_bef_humidity       1457 non-null   float64\n",
      " 6   hour_bef_visibility     1457 non-null   float64\n",
      " 7   hour_bef_ozone          1383 non-null   float64\n",
      " 8   hour_bef_pm10           1369 non-null   float64\n",
      " 9   hour_bef_pm2.5          1342 non-null   float64\n",
      " 10  count                   1459 non-null   float64\n",
      "dtypes: float64(9), int64(2)\n",
      "memory usage: 125.5 KB\n"
     ]
    }
   ],
   "source": [
    "train.info()\n",
    "# test.info()\n",
    "# 머신러닝의 모델들은 입력값에 결측값이 있으면 오류가 생길 수 있음. 결측값들을 사전에 채워줘야 함."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>hour</th>\n",
       "      <th>hour_bef_temperature</th>\n",
       "      <th>hour_bef_precipitation</th>\n",
       "      <th>hour_bef_windspeed</th>\n",
       "      <th>hour_bef_humidity</th>\n",
       "      <th>hour_bef_visibility</th>\n",
       "      <th>hour_bef_ozone</th>\n",
       "      <th>hour_bef_pm10</th>\n",
       "      <th>hour_bef_pm2.5</th>\n",
       "      <th>count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1459.000000</td>\n",
       "      <td>1459.000000</td>\n",
       "      <td>1457.000000</td>\n",
       "      <td>1457.000000</td>\n",
       "      <td>1450.000000</td>\n",
       "      <td>1457.000000</td>\n",
       "      <td>1457.000000</td>\n",
       "      <td>1383.000000</td>\n",
       "      <td>1369.000000</td>\n",
       "      <td>1342.000000</td>\n",
       "      <td>1459.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1105.914325</td>\n",
       "      <td>11.493489</td>\n",
       "      <td>16.717433</td>\n",
       "      <td>0.031572</td>\n",
       "      <td>2.479034</td>\n",
       "      <td>52.231297</td>\n",
       "      <td>1405.216884</td>\n",
       "      <td>0.039149</td>\n",
       "      <td>57.168736</td>\n",
       "      <td>30.327124</td>\n",
       "      <td>108.563400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>631.338681</td>\n",
       "      <td>6.922790</td>\n",
       "      <td>5.239150</td>\n",
       "      <td>0.174917</td>\n",
       "      <td>1.378265</td>\n",
       "      <td>20.370387</td>\n",
       "      <td>583.131708</td>\n",
       "      <td>0.019509</td>\n",
       "      <td>31.771019</td>\n",
       "      <td>14.713252</td>\n",
       "      <td>82.631733</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.100000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>78.000000</td>\n",
       "      <td>0.003000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>555.500000</td>\n",
       "      <td>5.500000</td>\n",
       "      <td>12.800000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.400000</td>\n",
       "      <td>36.000000</td>\n",
       "      <td>879.000000</td>\n",
       "      <td>0.025500</td>\n",
       "      <td>36.000000</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>37.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1115.000000</td>\n",
       "      <td>11.000000</td>\n",
       "      <td>16.600000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.300000</td>\n",
       "      <td>51.000000</td>\n",
       "      <td>1577.000000</td>\n",
       "      <td>0.039000</td>\n",
       "      <td>51.000000</td>\n",
       "      <td>26.000000</td>\n",
       "      <td>96.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1651.000000</td>\n",
       "      <td>17.500000</td>\n",
       "      <td>20.100000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.400000</td>\n",
       "      <td>69.000000</td>\n",
       "      <td>1994.000000</td>\n",
       "      <td>0.052000</td>\n",
       "      <td>69.000000</td>\n",
       "      <td>37.000000</td>\n",
       "      <td>150.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>2179.000000</td>\n",
       "      <td>23.000000</td>\n",
       "      <td>30.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>99.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>0.125000</td>\n",
       "      <td>269.000000</td>\n",
       "      <td>90.000000</td>\n",
       "      <td>431.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                id         hour  hour_bef_temperature  hour_bef_precipitation  \\\n",
       "count  1459.000000  1459.000000           1457.000000             1457.000000   \n",
       "mean   1105.914325    11.493489             16.717433                0.031572   \n",
       "std     631.338681     6.922790              5.239150                0.174917   \n",
       "min       3.000000     0.000000              3.100000                0.000000   \n",
       "25%     555.500000     5.500000             12.800000                0.000000   \n",
       "50%    1115.000000    11.000000             16.600000                0.000000   \n",
       "75%    1651.000000    17.500000             20.100000                0.000000   \n",
       "max    2179.000000    23.000000             30.000000                1.000000   \n",
       "\n",
       "       hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  \\\n",
       "count         1450.000000        1457.000000          1457.000000   \n",
       "mean             2.479034          52.231297          1405.216884   \n",
       "std              1.378265          20.370387           583.131708   \n",
       "min              0.000000           7.000000            78.000000   \n",
       "25%              1.400000          36.000000           879.000000   \n",
       "50%              2.300000          51.000000          1577.000000   \n",
       "75%              3.400000          69.000000          1994.000000   \n",
       "max              8.000000          99.000000          2000.000000   \n",
       "\n",
       "       hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5        count  \n",
       "count     1383.000000    1369.000000     1342.000000  1459.000000  \n",
       "mean         0.039149      57.168736       30.327124   108.563400  \n",
       "std          0.019509      31.771019       14.713252    82.631733  \n",
       "min          0.003000       9.000000        8.000000     1.000000  \n",
       "25%          0.025500      36.000000       20.000000    37.000000  \n",
       "50%          0.039000      51.000000       26.000000    96.000000  \n",
       "75%          0.052000      69.000000       37.000000   150.000000  \n",
       "max          0.125000     269.000000       90.000000   431.000000  "
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.describe()\n",
    "# test.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='hour'>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "train.groupby('hour').mean()['count'].plot()    # 시간별 평균 대여량 산출"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.plot(train.groupby('hour').mean()['count'], 'o-')\n",
    "plt.grid()    # 보조선 추가"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'count')"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(train.groupby('hour').mean()['count'], 'o-')\n",
    "plt.grid()\n",
    "\n",
    "plt.title('count by hours', fontsize = 15)\n",
    "plt.xlabel('hour', fontsize = 15)\n",
    "plt.ylabel('count', fontsize = 15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(18, 120, 'leave work')"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(train.groupby('hour').mean()['count'], 'o-')\n",
    "plt.grid()\n",
    "\n",
    "plt.title('count by hours', fontsize = 15)\n",
    "plt.xlabel('hour', fontsize = 15)\n",
    "plt.ylabel('count', fontsize = 15)\n",
    "\n",
    "plt.axvline(8, color = 'r')\n",
    "plt.axvline(18, color = 'r')    # 빨간선 표시\n",
    "\n",
    "plt.text(8, 120, 'go work', fontsize = 10)\n",
    "plt.text(18, 120, 'leave work', fontsize = 10)    # 텍스트 표시"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns    # 상관계수 한눈에 알아보기 위해 유용한 라이브러리 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# count와 상관계수가 높은 변수들을 골라서 모델을 생성한다.\n",
    "plt.figure(figsize=(10, 10))\n",
    "sns.heatmap(train.corr(), annot = True)    # annot : 수치도 표시\n",
    "\n",
    "# 3가지로 모델링 해보자 (시간, 온도, 풍속)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 데이터 전처리\n",
    "● 데이터 프레임을 최적의 형태로 만들어 주는 과정. 결측치에 관심을 가져야함."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "id                          0\n",
       "hour                        0\n",
       "hour_bef_temperature        2\n",
       "hour_bef_precipitation      2\n",
       "hour_bef_windspeed          9\n",
       "hour_bef_humidity           2\n",
       "hour_bef_visibility         2\n",
       "hour_bef_ozone             76\n",
       "hour_bef_pm10              90\n",
       "hour_bef_pm2.5            117\n",
       "count                       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.isna().sum()\n",
    "\n",
    "# 결측치면 True, 아니면 False 반환\n",
    "# 온도, 풍속 변수에 결측치 2개, 9개 존재"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>hour</th>\n",
       "      <th>hour_bef_temperature</th>\n",
       "      <th>hour_bef_precipitation</th>\n",
       "      <th>hour_bef_windspeed</th>\n",
       "      <th>hour_bef_humidity</th>\n",
       "      <th>hour_bef_visibility</th>\n",
       "      <th>hour_bef_ozone</th>\n",
       "      <th>hour_bef_pm10</th>\n",
       "      <th>hour_bef_pm2.5</th>\n",
       "      <th>count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>33</td>\n",
       "      <td>13</td>\n",
       "      <td>22.6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>41.0</td>\n",
       "      <td>987.0</td>\n",
       "      <td>0.046</td>\n",
       "      <td>64.0</td>\n",
       "      <td>39.0</td>\n",
       "      <td>208.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>244</th>\n",
       "      <td>381</td>\n",
       "      <td>1</td>\n",
       "      <td>14.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>55.0</td>\n",
       "      <td>1992.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>38.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>260</th>\n",
       "      <td>404</td>\n",
       "      <td>3</td>\n",
       "      <td>14.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>50.0</td>\n",
       "      <td>2000.0</td>\n",
       "      <td>0.049</td>\n",
       "      <td>35.0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>17.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>376</th>\n",
       "      <td>570</td>\n",
       "      <td>0</td>\n",
       "      <td>14.3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>49.0</td>\n",
       "      <td>2000.0</td>\n",
       "      <td>0.044</td>\n",
       "      <td>37.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>58.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>780</th>\n",
       "      <td>1196</td>\n",
       "      <td>20</td>\n",
       "      <td>16.5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>31.0</td>\n",
       "      <td>2000.0</td>\n",
       "      <td>0.058</td>\n",
       "      <td>39.0</td>\n",
       "      <td>18.0</td>\n",
       "      <td>181.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>934</th>\n",
       "      <td>1420</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>39.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1035</th>\n",
       "      <td>1553</td>\n",
       "      <td>18</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1138</th>\n",
       "      <td>1717</td>\n",
       "      <td>12</td>\n",
       "      <td>21.4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>44.0</td>\n",
       "      <td>1375.0</td>\n",
       "      <td>0.044</td>\n",
       "      <td>61.0</td>\n",
       "      <td>37.0</td>\n",
       "      <td>116.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1229</th>\n",
       "      <td>1855</td>\n",
       "      <td>2</td>\n",
       "      <td>14.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>52.0</td>\n",
       "      <td>2000.0</td>\n",
       "      <td>0.044</td>\n",
       "      <td>37.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>20.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        id  hour  hour_bef_temperature  hour_bef_precipitation  \\\n",
       "18      33    13                  22.6                     0.0   \n",
       "244    381     1                  14.1                     0.0   \n",
       "260    404     3                  14.0                     0.0   \n",
       "376    570     0                  14.3                     0.0   \n",
       "780   1196    20                  16.5                     0.0   \n",
       "934   1420     0                   NaN                     NaN   \n",
       "1035  1553    18                   NaN                     NaN   \n",
       "1138  1717    12                  21.4                     0.0   \n",
       "1229  1855     2                  14.0                     0.0   \n",
       "\n",
       "      hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  \\\n",
       "18                   NaN               41.0                987.0   \n",
       "244                  NaN               55.0               1992.0   \n",
       "260                  NaN               50.0               2000.0   \n",
       "376                  NaN               49.0               2000.0   \n",
       "780                  NaN               31.0               2000.0   \n",
       "934                  NaN                NaN                  NaN   \n",
       "1035                 NaN                NaN                  NaN   \n",
       "1138                 NaN               44.0               1375.0   \n",
       "1229                 NaN               52.0               2000.0   \n",
       "\n",
       "      hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5  count  \n",
       "18             0.046           64.0            39.0  208.0  \n",
       "244              NaN            NaN             NaN   38.0  \n",
       "260            0.049           35.0            22.0   17.0  \n",
       "376            0.044           37.0            20.0   58.0  \n",
       "780            0.058           39.0            18.0  181.0  \n",
       "934              NaN            NaN             NaN   39.0  \n",
       "1035             NaN            NaN             NaN    1.0  \n",
       "1138           0.044           61.0            37.0  116.0  \n",
       "1229           0.044           37.0            20.0   20.0  "
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train[train['hour_bef_temperature'].isna()]\n",
    "train[train['hour_bef_windspeed'].isna()]\n",
    "\n",
    "# 자정과 18시에 결측치 존재"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.lines.Line2D at 0x203391f78e0>"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 온도 변화\n",
    "\n",
    "train.groupby('hour').mean()['hour_bef_temperature'].plot()\n",
    "plt.axhline(train.groupby('hour').mean()['hour_bef_temperature'].mean())\n",
    "\n",
    "# 0시의 평균온도와와 전체 평균온도로 대체한다면, 모델링 결과에 차이가 크게 나타날 것임을 알았음.\n",
    "# 전체 평균온도를 넣기보다는 각 시간별 평균온도로 대체함으로써 시간대별 경향성을 반영해야겠다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "hour\n",
       "0     1.965517\n",
       "1     1.836667\n",
       "2     1.633333\n",
       "3     1.620000\n",
       "4     1.409836\n",
       "5     1.296721\n",
       "6     1.331148\n",
       "7     1.262295\n",
       "8     1.632787\n",
       "9     1.829508\n",
       "10    2.122951\n",
       "11    2.485246\n",
       "12    2.766667\n",
       "13    3.281356\n",
       "14    3.522951\n",
       "15    3.768852\n",
       "16    3.820000\n",
       "17    3.801667\n",
       "18    3.838333\n",
       "19    3.595082\n",
       "20    3.278333\n",
       "21    2.755000\n",
       "22    2.498361\n",
       "23    2.195082\n",
       "Name: hour_bef_windspeed, dtype: float64"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 시간대별 평균 온도 알아봄.\n",
    "\n",
    "train.groupby('hour').mean()['hour_bef_temperature']\n",
    "train.groupby('hour').mean()['hour_bef_windspeed']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 결측치 채워줄 때, 딕셔너리 사용\n",
    "\n",
    "train['hour_bef_temperature'].fillna({934:14.788136, 1035: 20.926667}, inplace = True)    # inplace = True 로 저장까지 완료\n",
    "train['hour_bef_windspeed'].fillna({18:3.281356, 244:1.836667, 260:1.620000, 376:1.965517, 780:3.278333, 934:1.965517, 1035:3.838333, 1138:2.766667, 1229:1.633333}, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "train.isna().sum()    # 전체적인 결측치 확인\n",
    "\n",
    "test[test['hour_bef_temperature'].isna()]    # 해당 칼럼의 결측치가 포함된 열 확인\n",
    "test['hour_bef_temperature'].fillna(19.704918, inplace = True)    # train의 시간대별 평균온도로 채워줌\n",
    "\n",
    "test[test['hour_bef_windspeed'].isna()]     # 해당 칼럼의 결측치가 포함된 열 확인\n",
    "test['hour_bef_windspeed'].fillna(3.595082, inplace = True)    # train의 시간대별 평균풍속으로 채워줌"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 모델링\n",
    "● train에서 시간, 온도, 풍속 변수를 뽑아 1시간 후의 따릉이 대여량 예측하는 모델"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "features = ['hour', 'hour_bef_temperature', 'hour_bef_windspeed']\n",
    "X_train = train[features]    # 모델 학습 시 사용\n",
    "y_train = train['count']    # 모델 학습 시 사용\n",
    "X_test = test[features]    # 모델의 성능 테스트"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1459, 3)\n",
      "(1459,)\n",
      "(715, 3)\n"
     ]
    }
   ],
   "source": [
    "print(X_train.shape)\n",
    "print(y_train.shape)\n",
    "print(X_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 랜덤 포레스트 모형 설정\n",
    "\n",
    "model100 = RandomForestRegressor(n_estimators=100, random_state=0)\n",
    "model100_5 = RandomForestRegressor(n_estimators=100, max_depth = 5, random_state=0)\n",
    "model200 = RandomForestRegressor(n_estimators=200)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor(n_estimators=200)"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 모델 학습\n",
    "\n",
    "model100.fit(X_train, y_train)\n",
    "model100_5.fit(X_train, y_train)\n",
    "model200.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 모델 예측\n",
    "\n",
    "ypred1 = model100.predict(X_test)\n",
    "ypred2 = model100_5.predict(X_test)\n",
    "ypred3 = model200.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "# submission 파일에 예측한 결과 저장\n",
    "\n",
    "submission['count'] = ypred1\n",
    "submission.to_csv('model100.csv', index = False)\n",
    "\n",
    "submission['count'] = ypred2\n",
    "submission.to_csv('model100_5.csv', index = False)\n",
    "\n",
    "submission['count'] = ypred3\n",
    "submission.to_csv('model200.csv', index = False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
