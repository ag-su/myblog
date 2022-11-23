---  
layout: post   
title: Transcription || [Kaggle] Credit card Fraud Detection
image: 01.credit_thumbnail.png
tags:  
categories: transcription
---

**[필사 코드 링크]**  
-[Credit Fraud || Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/notebook)

**[데이터셋 링크]**  
-[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 캐글에서 Credit Card Fraud Detection 데이터 불러오기   



```python
# install kaggle 
# !pip install kaggle --upgrade
```


```python
# API Token 업로드
from google.colab import files
import pandas as pd
files.upload()

# json 파일 ~/.kaggle로 이동시키기
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# Permission Warning 방지
!chmod 600 ~/.kaggle/kaggle.json

# 데이터셋 다운로드
!kaggle datasets download -d mlg-ulb/creditcardfraud
!unzip  creditcardfraud.zip
df = pd.read_csv('creditcard.csv')
```

<br>
<br>
       
    

**Introduction**  
- 우리는 다양한 예측 모델을 사용하여 거래가 정상적인 지불인지 사기인지 탐지하는 데 얼마나 정확한지 확인한다. 데이터 세트에서 볼 수 있듯, 개인 정보 보호의 이유로 인해  변수의 크기가 표준화되었고, 변수의 이름도 V1~V28로 되어있다. 하지만 우리는 이 데이터셋을 사용하여 몇 가지 중요한 측면을 분석할 수 있다.

<br>

**목표**  
- 우리에게 제공된 "작은" 데이터의 작은 분포를 이해한다.
- "사기" 트랜잭션과 "비사기" 트랜잭션의 50/50 하위 데이터 프레임 비율을 만든다. (Near Miss 알고리즘)
- 사용할 분류기를 결정하고 정확도가 높은 분류기를 결정한다.
- 신경망(neural network)을 만들고 정확도를 우리의 최고 분류기와 비교한다.
- 불균형 데이터셋에서 흔히 발생하는 실수를 이해한다.

<br>

**개요**  
- 1.데이터 이해 
  - 1.1 데이터에 대한 정보 수집


- 2.전처리  
  - 2.1 표준화(Scaling)와 분포(Distributing)
  - 2.2 데이터 분할 (Spliting the Data)


- 3.랜덤 언더 샘플링(Under Sampling) 및 오버 샘플링(Over Sampling)  
  - 3.1 분포(Distributing) 및 상관 관계(Correlating)
  - 3.2 이상 탐지(Anomaly Detection)
  - 3.3 차원 축소 및 클러스터링(t-SNE)
  - 3.4 Classifiers
  - 3.5 로지스틱 회귀 분석
  - 3.6 SMOTE를 사용한 오버샘플링(Over Smpling)


- 4.테스팅  
  - 4.1 로지스틱 회귀 분석을 사용한 검정(Testing)
  - 4.2 신경망 테스팅(언더샘플링 vs 오버샘플링)

# 1. 데이터 이해

## 1.1 데이터에 대한 정보 수집
우리가 가장 먼저 해야 할 일은 데이터에 대한 기본적인 감각을 익히는 것이다. 개인 정보 보호 상의 이유로, 거래 및 금액을 제외하고 다른 컬럼에 대한 정보는 알 수 없다. 우리가 아는 유일한 것은 그 컬럼들은 이미 `표준화`되어있다는 점이다. 

**변수에 적용된 세부적인 기술**  
- PCA 변환: 데이터의 설명에 따르면, time, amount 변수를 제외한 모든 변수는 PCA 변환(Dimensionality Reduction technique)을 거쳤다.

- 스케일링: PCA 변환을 구현하려면 변수를 미리 스케일링해야 한다. 

library import 


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
```

- 데이터 확인 


```python
df.head()
```





  <div id="df-1a8212d1-eee1-4c64-b0d4-64b317ab7d1d">
    <div class="colab-df-container">
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1a8212d1-eee1-4c64-b0d4-64b317ab7d1d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1a8212d1-eee1-4c64-b0d4-64b317ab7d1d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1a8212d1-eee1-4c64-b0d4-64b317ab7d1d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




- 기초 통계량 확인 


```python
df.describe()
```





  <div id="df-cf989260-5366-4a5d-baf3-2023c9cf1b66">
    <div class="colab-df-container">
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284807.000000</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>...</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>284807.000000</td>
      <td>284807.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>94813.859575</td>
      <td>1.168375e-15</td>
      <td>3.416908e-16</td>
      <td>-1.379537e-15</td>
      <td>2.074095e-15</td>
      <td>9.604066e-16</td>
      <td>1.487313e-15</td>
      <td>-5.556467e-16</td>
      <td>1.213481e-16</td>
      <td>-2.406331e-15</td>
      <td>...</td>
      <td>1.654067e-16</td>
      <td>-3.568593e-16</td>
      <td>2.578648e-16</td>
      <td>4.473266e-15</td>
      <td>5.340915e-16</td>
      <td>1.683437e-15</td>
      <td>-3.660091e-16</td>
      <td>-1.227390e-16</td>
      <td>88.349619</td>
      <td>0.001727</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47488.145955</td>
      <td>1.958696e+00</td>
      <td>1.651309e+00</td>
      <td>1.516255e+00</td>
      <td>1.415869e+00</td>
      <td>1.380247e+00</td>
      <td>1.332271e+00</td>
      <td>1.237094e+00</td>
      <td>1.194353e+00</td>
      <td>1.098632e+00</td>
      <td>...</td>
      <td>7.345240e-01</td>
      <td>7.257016e-01</td>
      <td>6.244603e-01</td>
      <td>6.056471e-01</td>
      <td>5.212781e-01</td>
      <td>4.822270e-01</td>
      <td>4.036325e-01</td>
      <td>3.300833e-01</td>
      <td>250.120109</td>
      <td>0.041527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-5.640751e+01</td>
      <td>-7.271573e+01</td>
      <td>-4.832559e+01</td>
      <td>-5.683171e+00</td>
      <td>-1.137433e+02</td>
      <td>-2.616051e+01</td>
      <td>-4.355724e+01</td>
      <td>-7.321672e+01</td>
      <td>-1.343407e+01</td>
      <td>...</td>
      <td>-3.483038e+01</td>
      <td>-1.093314e+01</td>
      <td>-4.480774e+01</td>
      <td>-2.836627e+00</td>
      <td>-1.029540e+01</td>
      <td>-2.604551e+00</td>
      <td>-2.256568e+01</td>
      <td>-1.543008e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>54201.500000</td>
      <td>-9.203734e-01</td>
      <td>-5.985499e-01</td>
      <td>-8.903648e-01</td>
      <td>-8.486401e-01</td>
      <td>-6.915971e-01</td>
      <td>-7.682956e-01</td>
      <td>-5.540759e-01</td>
      <td>-2.086297e-01</td>
      <td>-6.430976e-01</td>
      <td>...</td>
      <td>-2.283949e-01</td>
      <td>-5.423504e-01</td>
      <td>-1.618463e-01</td>
      <td>-3.545861e-01</td>
      <td>-3.171451e-01</td>
      <td>-3.269839e-01</td>
      <td>-7.083953e-02</td>
      <td>-5.295979e-02</td>
      <td>5.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>84692.000000</td>
      <td>1.810880e-02</td>
      <td>6.548556e-02</td>
      <td>1.798463e-01</td>
      <td>-1.984653e-02</td>
      <td>-5.433583e-02</td>
      <td>-2.741871e-01</td>
      <td>4.010308e-02</td>
      <td>2.235804e-02</td>
      <td>-5.142873e-02</td>
      <td>...</td>
      <td>-2.945017e-02</td>
      <td>6.781943e-03</td>
      <td>-1.119293e-02</td>
      <td>4.097606e-02</td>
      <td>1.659350e-02</td>
      <td>-5.213911e-02</td>
      <td>1.342146e-03</td>
      <td>1.124383e-02</td>
      <td>22.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>139320.500000</td>
      <td>1.315642e+00</td>
      <td>8.037239e-01</td>
      <td>1.027196e+00</td>
      <td>7.433413e-01</td>
      <td>6.119264e-01</td>
      <td>3.985649e-01</td>
      <td>5.704361e-01</td>
      <td>3.273459e-01</td>
      <td>5.971390e-01</td>
      <td>...</td>
      <td>1.863772e-01</td>
      <td>5.285536e-01</td>
      <td>1.476421e-01</td>
      <td>4.395266e-01</td>
      <td>3.507156e-01</td>
      <td>2.409522e-01</td>
      <td>9.104512e-02</td>
      <td>7.827995e-02</td>
      <td>77.165000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>172792.000000</td>
      <td>2.454930e+00</td>
      <td>2.205773e+01</td>
      <td>9.382558e+00</td>
      <td>1.687534e+01</td>
      <td>3.480167e+01</td>
      <td>7.330163e+01</td>
      <td>1.205895e+02</td>
      <td>2.000721e+01</td>
      <td>1.559499e+01</td>
      <td>...</td>
      <td>2.720284e+01</td>
      <td>1.050309e+01</td>
      <td>2.252841e+01</td>
      <td>4.584549e+00</td>
      <td>7.519589e+00</td>
      <td>3.517346e+00</td>
      <td>3.161220e+01</td>
      <td>3.384781e+01</td>
      <td>25691.160000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-cf989260-5366-4a5d-baf3-2023c9cf1b66')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-cf989260-5366-4a5d-baf3-2023c9cf1b66 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cf989260-5366-4a5d-baf3-2023c9cf1b66');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




- 결측값 확인 


```python
df.isnull().sum().max()
```




    0



- 컬럼 확인 


```python
df.columns
```




    Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
           'Class'],
          dtype='object')



- class 비율 확인하기 

데이터 불균형 문제가 심각하므로 이후에 이 문제를 해결해야한다. 


```python
print(f"No Frauds: {round(df['Class'].value_counts()[0]/len(df)*100, 2)}% of the dataset")
print(f"Frauds: {round(df['Class'].value_counts()[1]/len(df)*100, 2)}% of the dataset")
```

    No Frauds: 99.83% of the dataset
    Frauds: 0.17% of the dataset
    


```python
colors = ['#0101DF', '#DF0101']
sns.countplot('Class', data=df, palette=colors)
plt.title('Class Distributions \n (0: No Fraud | 1: Fraud)', fontsize=14)
```




    Text(0.5, 1.0, 'Class Distributions \n (0: No Fraud | 1: Fraud)')




    
![png]({{site.baseurl}}/images/output_20_1.png)
    


분포를 보면 컬럼들이 얼마나 왜곡되어 있는지 알 수 있고, 향후 이 글에서 구현될 기술로 분포의 왜곡을 줄일 수 있다.

- Amount, time 분포 확인 


```python
fig, ax = plt.subplots(1, 2, figsize=(18, 4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])
```




    (0.0, 172792.0)




    
![png]({{site.baseurl}}/images/output_23_1.png)
    


**요약**  
- 거래금액이 상대적으로 적다. 모든 금액의 평균은 대략 88달러였다.
- "Null" 값이 없으므로 결측값 처리 방법을 찾을 필요가 없다.
- 거래의 대부분은 **비사기**(99.83%)였고, 나머지 0.17%가 **사기**였다.

------

# 2. 전처리 (Preprocessing) 




## 2.1 표준화(Scaling)와 분포(Distributing)
이번 단계에서는 먼저 `Time`, `Amount` 컬럼을 표준화한다. 두 컬럼도 다른 컬럼들 처럼 표준화 할 필요가 있다. 한편, class의 불균형 문제를 해결하기 위해 데이터 프레임의 `하위 샘플(sub sample)`도 생성해야 하며, 이는 알고리즘이 사기 여부를 결정하는 패턴을 더 잘 이해하는데 도움을 줄 수 있다. 


**하위 샘플(sub sample) 이란?**
- 우리의 하위 샘플은 사기:비사기 비율이 50:50인 데이터 프레임이 될 것이다. 즉, 우리의 하위 샘플은 동일한 양의 class를 갖는다. 


**하위 샘플을 생성하는 이유?**
- 이 글의 시작 부분에서 원본 데이터 프레임의 불균형이 심하다는 것을 알았다. 원본 데이터 프레임을 사용하면 다음과 같은 문제가 발생한다.
- **과적합:** 우리의 분류 모델은 대부분의 경우 사기가 없다고 추측할 것이다. 우리가 우리 모델을 위해 원하는 것은 사기가 발생했을 때 확실한 판단을 하도록 하는 것이다. 
- **잘못된 상관 관계:** 비록 "V" 변수들이 무엇을 의미하는지 모르지만, 클래스와 "V" 변수 사이의 진정한 상관관계를 볼 수 없는 불균형 데이터 프레임을 가지면, 이러한 각 변수가 결과에 어떻게 영향을 미치는지 아는데 유용할 것이다.


**요약**
- scaled Amount, scaled Time 컬럼을 추가한다.
- 데이터셋에는 492건의 `사기` 데이터가 있으므로 무작위로 492건의 `비사기` 데이터를 얻어 새로운 하위 데이터 프레임을 만든다.
- 492건의 `사기` 및 `비사기` 데이터를 합쳐 새로운 하위 샘플을 만든다.

<br>

- scaling 


```python
from sklearn.preprocessing import StandardScaler, RobustScaler

#  RobustScaler가 이상치가 더 적게 발생한다. (less prone to outliers)

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)
```


```python
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

df.head()
```





  <div id="df-c3b4dd7b-9091-4502-ad16-f014986b7c56">
    <div class="colab-df-container">
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
      <th>scaled_amount</th>
      <th>scaled_time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>...</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.783274</td>
      <td>-0.994983</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>...</td>
      <td>0.251412</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.269825</td>
      <td>-0.994983</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>...</td>
      <td>-0.069083</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.983721</td>
      <td>-0.994972</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>...</td>
      <td>0.524980</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.418291</td>
      <td>-0.994972</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>...</td>
      <td>-0.208038</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.670579</td>
      <td>-0.994960</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>...</td>
      <td>0.408542</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c3b4dd7b-9091-4502-ad16-f014986b7c56')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c3b4dd7b-9091-4502-ad16-f014986b7c56 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c3b4dd7b-9091-4502-ad16-f014986b7c56');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## 2.2 데이터 분할 (Spliting the Data)
랜덤 언더샘플링 기법을 진행하기 전 원본 데이터 프레임을 분리해야 한다. 테스트로 랜덤 언더샘플링, 오버샘플링 기술을 구현할 때 데이터를 분할한다. 하지만 우리는 이 기술들에 의해 생성된 테스트셋이 아닌 원본 데이터의 테스트셋으로 모델을 평가해야한다. 모델이 패턴을 감지할 수 있도록 언더샘플, 오버샘플 데이터프레임에 모델을 학습시키고, 원본 테스트셋에서 평가하는 것이 우리의 목표이기 때문이다. 


```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

print(f"No Frauds: {round(df['Class'].value_counts()[0]/len(df)*100, 2)}% of the dataset")
print(f"Frauds: {round(df['Class'].value_counts()[1]/len(df)*100, 2)}% of the dataset")
print()

X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

# train, test 분리 
for train_index, test_index in sss.split(X, y):
  print('Train:', train_index, 'Test:', test_index,)
  original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
  original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index] 

# array로 변환 
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values 

# train, test의 label 분포가 유사한지 확인한다. 
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distribution: \n')
print(train_counts_label / len(original_ytrain))
print(test_counts_label / len(original_ytest))
```

    No Frauds: 99.83% of the dataset
    Frauds: 0.17% of the dataset
    
    Train: [ 56953  56954  56955 ... 284804 284805 284806] Test: [    0     1     2 ... 59039 59380 59968]
    Train: [     0      1      2 ... 284804 284805 284806] Test: [ 56953  56954  56955 ... 113922 113923 113924]
    Train: [     0      1      2 ... 284804 284805 284806] Test: [113764 113925 113926 ... 170892 170893 170894]
    Train: [     0      1      2 ... 284804 284805 284806] Test: [166690 166692 167046 ... 227845 227846 227847]
    Train: [     0      1      2 ... 227845 227846 227847] Test: [227429 227701 227848 ... 284804 284805 284806]
    ----------------------------------------------------------------------------------------------------
    Label Distribution: 
    
    [0.99827076 0.00172924]
    [0.99827952 0.00172048]
    

\* 5-fold로 데이터셋을 나누는 for문에서 계속 덮어씌우도록 작성을 하고 결국 마지막 데이터셋만 사용한다. 뒤에서도 이러는데 이렇게 작성한 이유가 궁금하다.   
\* -> 댓글에서 같은 질문이 있는데 작성자님이 오류인 것 같고 혼란스럽다고 했지만 아직 해결은 안 난 것 같다.

# 3.랜덤 언더 샘플링(Under Sampling) 및 오버 샘플링(Over Sampling)

균형 잡힌 데이터셋을 생성하기 위해, 데이터를 제거하여 모델의 과적합을 방지하는 `"랜덤 언더샘플링 (Random Under-Sampling)"`을 구현한다. 


**단계**  
1) 우리의 클래스가 얼마나 불균형한지 결정한다. (클래스 컬럼에 `value_counts` 함수를 적용)  
2) 사기 거래의 개수에 맞춰 사기 492, 비사기 492건으로 만든다. (50:50) 이를 통해 클래스에 대한 50:50 비율의 하위 샘플을 갖게된다.  
3) 이 스크립트를 실행할 때마다 모델이 특정 정확도를 유지할 수 있는지 확인하기 위해 데이터를 섞는다.   


**참고**  
: 랜덤 언더샘플링 (Random Under-Sampling)의 주요 문제는 정보 손실이 크다는 것이다. 때문에 분류 모델이 원하는 만큼 정확하게 학습되지 않을 위험이 존재한다. (284,315개의 비사기 거래 중 492개의 비사기 거래만을 사용하게 된다.) 






```python
print(df['Class'].value_counts())

# 하위 샘플(sub sample) 생성 전 데이터 섞기 
df = df.sample(frac=1)

# 사기 class, 비사기 class 비율 50:50인 데이터프레임 생성 
fraud_df = df.loc[df['Class']==1]
non_fraud_df = df.loc[df['Class']==0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# 데이터프레임 섞기 
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()
```

    0    284315
    1       492
    Name: Class, dtype: int64
    





  <div id="df-4c7ff76c-5f04-4b68-b537-3ef506e437c7">
    <div class="colab-df-container">
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
      <th>scaled_amount</th>
      <th>scaled_time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>...</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>281645</th>
      <td>-0.279746</td>
      <td>1.006074</td>
      <td>0.669369</td>
      <td>0.802239</td>
      <td>-0.548859</td>
      <td>-0.461751</td>
      <td>1.103532</td>
      <td>-0.877397</td>
      <td>1.152531</td>
      <td>-0.446391</td>
      <td>...</td>
      <td>-0.052753</td>
      <td>-0.294856</td>
      <td>-0.573588</td>
      <td>0.208144</td>
      <td>0.651524</td>
      <td>-0.988035</td>
      <td>-0.034410</td>
      <td>-0.142753</td>
      <td>-0.182496</td>
      <td>0</td>
    </tr>
    <tr>
      <th>151006</th>
      <td>-0.293440</td>
      <td>0.113606</td>
      <td>-26.457745</td>
      <td>16.497472</td>
      <td>-30.177317</td>
      <td>8.904157</td>
      <td>-17.892600</td>
      <td>-1.227904</td>
      <td>-31.197329</td>
      <td>-11.438920</td>
      <td>...</td>
      <td>2.812241</td>
      <td>-8.755698</td>
      <td>3.460893</td>
      <td>0.896538</td>
      <td>0.254836</td>
      <td>-0.738097</td>
      <td>-0.966564</td>
      <td>-7.263482</td>
      <td>-1.324884</td>
      <td>1</td>
    </tr>
    <tr>
      <th>252962</th>
      <td>1.935304</td>
      <td>0.838179</td>
      <td>1.608340</td>
      <td>-0.833553</td>
      <td>-0.234511</td>
      <td>0.494109</td>
      <td>-1.133410</td>
      <td>-0.810632</td>
      <td>-0.350734</td>
      <td>-0.003276</td>
      <td>...</td>
      <td>0.028270</td>
      <td>0.049013</td>
      <td>-0.153223</td>
      <td>0.282484</td>
      <td>0.478331</td>
      <td>-0.554943</td>
      <td>-1.001841</td>
      <td>0.020994</td>
      <td>-0.005208</td>
      <td>0</td>
    </tr>
    <tr>
      <th>252124</th>
      <td>-0.296653</td>
      <td>0.833774</td>
      <td>-1.928613</td>
      <td>4.601506</td>
      <td>-7.124053</td>
      <td>5.716088</td>
      <td>1.026579</td>
      <td>-3.189073</td>
      <td>-2.261897</td>
      <td>1.185096</td>
      <td>...</td>
      <td>0.328796</td>
      <td>0.602291</td>
      <td>-0.541287</td>
      <td>-0.354639</td>
      <td>-0.701492</td>
      <td>-0.030973</td>
      <td>0.034070</td>
      <td>0.573393</td>
      <td>0.294686</td>
      <td>1</td>
    </tr>
    <tr>
      <th>56703</th>
      <td>-0.296793</td>
      <td>-0.436413</td>
      <td>1.176716</td>
      <td>0.557091</td>
      <td>-0.490800</td>
      <td>0.756424</td>
      <td>0.249192</td>
      <td>-0.781871</td>
      <td>0.228750</td>
      <td>-0.040840</td>
      <td>...</td>
      <td>-0.102772</td>
      <td>-0.062166</td>
      <td>-0.128168</td>
      <td>-0.040176</td>
      <td>0.110040</td>
      <td>0.437891</td>
      <td>0.368809</td>
      <td>-0.018287</td>
      <td>0.031173</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4c7ff76c-5f04-4b68-b537-3ef506e437c7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4c7ff76c-5f04-4b68-b537-3ef506e437c7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4c7ff76c-5f04-4b68-b537-3ef506e437c7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## 3.1 분포(Distributing) 및 상관 관계(Correlating)
이제 데이터 프레임의 균형을 올바르게 맞추었으므로, 분석 및 데이터 전처리를 더 진행할 수 있다. 


```python
print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))

sns.countplot('Class', data=new_df, palette=colors)
plt.title('Equally Distribution Classes', fontsize=14)
plt.show()
```

    Distribution of the Classes in the subsample dataset
    0    0.5
    1    0.5
    Name: Class, dtype: float64
    


    
![png]({{site.baseurl}}/images/output_37_1.png)
    


**상관 행렬(correlation Matrices)**
- 상관 행렬은 우리의 데이터를 이해하는 본질이다. 특정 거래가 사기인지 여부에 큰 영향을 미치는 변수가 있는지 분석해야 한다. 사기 거래와 관련하여 어떤 변수가 높은 양 또는 음의 상관관계를 가지고 있는지 확인하기 위해 올바른 데이터프레임 (하위 샘플)을 사용하는 것이 중요하다. 


**참고**  
- 우리는 상관 행렬에서 하위 샘플(sub sample)을 사용해야 한다. 그렇지 않으면 원본 데이터 프레임의 높은 클래스 불균형으로 인해 문제점이 생긴다. 원본 데이터 프레임을 사용한다면, 우리의 상관 행렬이 우리 클래스 사이의 높은 불균형에 의해 영향을 받을 것이다.  


```python
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 20))

corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)

sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('SubSample  Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()
```


    
![png]({{site.baseurl}}/images/output_39_0.png)
    


- boxplot 


```python
# 클래스와 음의 상관관계인 변수들의 boxplot: 변수의 값이 작을수록 사기 클래스일 가능성이 높다. 
f, axes = plt.subplots(ncols=4, figsize=(20, 4))

sns.boxplot(x='Class', y='V17', data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x='Class', y='V14', data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V14 vs Class Negative Correlation')

sns.boxplot(x='Class', y='V12', data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')

sns.boxplot(x='Class', y='V10', data=new_df, palette=colors, ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')
```




    Text(0.5, 1.0, 'V10 vs Class Negative Correlation')




    
![png]({{site.baseurl}}/images/output_41_1.png)
    



```python
# 클래스와 양의 상관관계인 변수들의 boxplot: 변수의 값이 클수록 사기 클래스일 가능성이 높다. 
f, axes = plt.subplots(ncols=4, figsize=(20, 4))

sns.boxplot(x='Class', y='V11', data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V11 vs Class Negative Correlation')

sns.boxplot(x='Class', y='V4', data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V4 vs Class Negative Correlation')

sns.boxplot(x='Class', y='V2', data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V2 vs Class Negative Correlation')

sns.boxplot(x='Class', y='V19', data=new_df, palette=colors, ax=axes[3])
axes[3].set_title('V19 vs Class Negative Correlation')
```




    Text(0.5, 1.0, 'V19 vs Class Negative Correlation')




    
![png]({{site.baseurl}}/images/output_42_1.png)
    


**요약 및 설명**  
- 음의 상관: V17, V14, V12 및 V10은 음의 상관 관계입니다. 이러한 값이 얼마나 낮을수록 최종 결과는 부정 거래일 가능성이 더 높습니다.
- 양의 상관: V2, V4, V11 및 V19는 양의 상관 관계에 있습니다. 이러한 값이 얼마나 높을수록 최종 결과는 부정 거래일 가능성이 더 높습니다.
- 상자 그림: 우리는 상자 그림을 사용하여 부정 행위 및 비부정 행위에서 이러한 특징의 분포를 더 잘 이해할 것이다.

## 3.2 이상 탐지(Anomaly Detection)
클래스와의 상관관계가 높은 변수에서 `"극한 이상치 (extreme outliers)`를 제거하는 것이 목표이다. 이는 우리 모델의 정확도에 긍정적인 영향을 미칠 것이다. 

<br>

**이상치 탐지 방법**
- Interquartile Range(IQR): 우리는 제3사분위수와 제1사분위수의 차이로 IQR를 계산한다. 우리의 목표는 제3사분위수와 제2사분위수를 초과하는 임계값을 만드는 것이다. 일부 인스턴스가 이 임계값을 초과할 경우 삭제될 것이다.
- Boxplots: 제1사분위수와 제3사분위수(제곱의 양쪽 끝)를 쉽게 볼 수 있을 뿐만 아니라 극단 특이치(하위 및 상위 극한을 벗어난 점)도 쉽게 볼 수 있다.


**Outlier Removal Tradeoff**  
- 특이치를 제거하기 위한 임계값이 어디까지인지 주의해야한다. 숫자(예: 1.5)에 (사분위수 범위)를 곱하여 임계값을 결정한다. 이 임계값이 높을수록 이상치는 더 적게 탐지되고(높은 수 ex: 3을 곱함) 이 임계값이 낮을수록 더 많은 특이치를 탐지한다.
- the tradeoff: 임계값이 낮을수록 이상치가 더 많이 제거되지만, 우리는 이상치보다 "극한 이상치"에 더 초점을 맞추도록 한다. 우리의 모델이 더 낮은 정확도를 갖게 될 정보 손실의 위험을 최소화하기 위함이다. 이 임계값을 사용하여 분류 모델의 정확도에 어떤 영향을 미치는지 확인할 수 있다. 


**요약**
- 분포 시각화: 먼저 일부 이상치를 제거하는 데 사용할 형상의 분포를 시각화 한다. 
- 임계값 결정: IQR을 곱하기 위해 사용할 숫자를 결정한 후(더 적은 이상치가 제거될수록) Q25 - 임계값(하한 극단 임계값)을 기판으로 하고 Q75 + 임계값(상한 극단 임계값)을 추가하여 상한 및 하한 임계값을 결정하는 작업을 진행한다.
- 조건부 드롭: 마지막으로 양쪽 극단에서 "임계값"을 초과하면 인스턴스가 제거된다는 조건부 드롭을 생성한다.
- 상자 그림 표현: 상자 그림을 통해 "극한 이상치"의 수가 상당한 양으로 감소했음을 시각화한다.


**참고** 
- 특이치 감소를 구현한 후 정확도가 3% 이상 향상되었다. 일부 특이치는 모델의 정확도를 왜곡할 수 있지만, 극단적인 양의 정보 손실을 피해야 합니다. 그렇지 않으면 모델이 적합하지 않을 위험이 있다.


```python
from scipy.stats import norm 

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class']==1].values 
sns.distplot(v14_fraud_dist, ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = new_df['V12'].loc[new_df['Class']==1].values 
sns.distplot(v12_fraud_dist, ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v10_fraud_dist = new_df['V10'].loc[new_df['Class']==1].values 
sns.distplot(v10_fraud_dist, ax=ax3, fit=norm, color='#C5B3F9')
ax1.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)
```




    Text(0.5, 1.0, 'V10 Distribution \n (Fraud Transactions)')




    
![png]({{site.baseurl}}/images/output_45_1.png)
    


V14는 변수 V12 및 V10에 비해 가우스 분포를 갖는 유일한 변수이다.



```python
# V14 변수의 이상치 제거 (가장 높은 음의 상관관계를 가진 변수)
v14_fraud = new_df['V14'].loc[new_df['Class']==1].values 
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr{}'.format(v14_iqr))

v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off  
print('Cut off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print("Feature V14 Outliers for Fraud Cases: {}".format(len(outliers)))
print("V14 outliers:{}".format(outliers))

new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))
print('-' * 100)


# V12 변수의 이상치 제거 
v12_fraud = new_df['V12'].loc[new_df['Class']==1].values 
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v12_iqr = q75 - q25
print('iqr{}'.format(v12_iqr))

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off  
print('Cut off: {}'.format(v12_cut_off))
print('V12 Lower: {}'.format(v12_lower))
print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print("Feature V12 Outliers for Fraud Cases: {}".format(len(outliers)))
print("V12 outliers:{}".format(outliers))

new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))
print('-' * 100)


# V10 변수의 이상치 제거 
v10_fraud = new_df['V10'].loc[new_df['Class']==1].values 
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v10_iqr = q75 - q25
print('iqr{}'.format(v10_iqr))

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off  
print('Cut off: {}'.format(v10_cut_off))
print('V10 Lower: {}'.format(v10_lower))
print('V10 Upper: {}'.format(v10_upper))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print("Feature V10 Outliers for Fraud Cases: {}".format(len(outliers)))
print("V10 outliers:{}".format(outliers))

new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))
```

    Quartile 25: -9.692722964972386 | Quartile 75: -4.282820849486865
    iqr5.409902115485521
    Cut off: 8.114853173228282
    V14 Lower: -17.807576138200666
    V14 Upper: 3.8320323237414167
    Feature V14 Outliers for Fraud Cases: 4
    V14 outliers:[-18.0499976898594, -18.8220867423816, -18.4937733551053, -19.2143254902614]
    Number of Instances after outliers removal: 980
    ----------------------------------------------------------------------------------------------------
    Quartile 25: -8.67303320439115 | Quartile 75: -2.893030568676315
    iqr5.780002635714835
    Cut off: 8.670003953572252
    V12 Lower: -17.3430371579634
    V12 Upper: 5.776973384895937
    Feature V12 Outliers for Fraud Cases: 4
    V12 outliers:[-18.4311310279993, -18.5536970096458, -18.6837146333443, -18.0475965708216]
    Number of Instances after outliers removal: 976
    ----------------------------------------------------------------------------------------------------
    Quartile 25: -7.466658535821847 | Quartile 75: -2.5118611381562523
    iqr4.954797397665595
    Cut off: 7.432196096498393
    V10 Lower: -14.89885463232024
    V10 Upper: 4.92033495834214
    Feature V10 Outliers for Fraud Cases: 27
    V10 outliers:[-22.1870885620007, -24.5882624372475, -15.1241628144947, -19.836148851696, -15.5637913387301, -15.3460988468775, -16.3035376590131, -23.2282548357516, -16.2556117491401, -22.1870885620007, -16.7460441053944, -18.2711681738888, -18.9132433348732, -15.2399619587112, -24.4031849699728, -16.6496281595399, -17.1415136412892, -15.5637913387301, -22.1870885620007, -14.9246547735487, -20.9491915543611, -15.1237521803455, -16.6011969664137, -15.2399619587112, -14.9246547735487, -22.1870885620007, -15.2318333653018]
    Number of Instances after outliers removal: 945
    


```python
f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))

colors = ['#B3F9C5', '#f9c5b3']
# Boxplots with outliers removed
# Feature V14
sns.boxplot(x="Class", y="V14", data=new_df,ax=ax1, palette=colors)
ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)
ax1.annotate('Fewer extreme \n outliers', xy=(0.98, -17.8), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

# Feature 12
sns.boxplot(x="Class", y="V12", data=new_df, ax=ax2, palette=colors)
ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)
ax2.annotate('Fewer extreme \n outliers', xy=(0.98, -17.3), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

# Feature V10
sns.boxplot(x="Class", y="V10", data=new_df, ax=ax3, palette=colors)
ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)
ax3.annotate('Fewer extreme \n outliers', xy=(0.98, -14.8), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)


plt.show()
```


    
![png]({{site.baseurl}}/images/output_48_0.png)
    


## 3.3 차원 축소 및 클러스터링(t-SNE)
**t-SNE의 이해**  
: 이 알고리즘을 이해하려면 다음의 용어들을 이해해야 한다. 
- 유클리드 거리
- 조건부 확률 
- 정규 분포 및 t-분포

\* Joshua Starmer가 t-SNE에 대하여 명확하게 설명한 간단한 안내 비디오를 [StatQuest: t-SNE, Clearly Explained](https://www.youtube.com/watch?v=NEaUSP4YerM) 에서 볼 수 있다. 

**요약**  
- t-SNE 알고리즘은 우리의 데이터셋에서 사기 및 비사기 클래스를 꽤 정확하게 클러스터링할 수 있다.
- 하위 샘플(sub sample)은 매우 작지만, t-SNE 알고리즘은 모든 시나리오에서 클러스터를 꽤 정확하게 감지할 수 있다. (t-SNE를 실행하기 전에 데이터 세트를 섞는다.)
- 이는 추가 예측 모델이 사기, 비사기 클래스를 분리하는 데 있어 상당히 우수한 성능을 발휘할 것임을 시사한다. 


```python
# 언더샘플 데이터인 new_df를 사용한다. (더 적은 데이터임을 기억하자.)
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# t-SNE Implementation 
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print('t-SNE took {:.2} s'.format(t1-t0))

# PCA Implementation 
t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print("PCA took {:.2} s".format(t1-t0))

# TruncateedSVD 
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
t1 = time.time()
print("Truncated SVD took {:.2} s".format(t1-t0))
```

    t-SNE took 1.4e+01 s
    PCA took 0.017 s
    Truncated SVD took 0.0072 s
    


```python
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)

ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()
```


    
![png]({{site.baseurl}}/images/output_51_0.png)
    


## 3.4 Classifiers
네 가지 유형의 분류기를 학습하고, 사기 거래 탐지에 효과적인 분류기를 결정한다. train, test 데이터셋, label을 불리한다. 


**요약**
- **로지스틱 회귀 분석** 분류기는 대부분의 경우 다른 세 분류기보다 정확하다. (Logistic Regression(로지스틱 회귀 분석)을 추가로 분석한다.)
- **GridSearchCV**는 분류기에 대한 최상의 예측 점수를 제공하는 매개 변수를 결정하는 데 사용된다.
- 로지스틱 회귀 분석에서는 가장 좋은 ROC(Receiving Operating Characteristic)를 갖는다. 즉, 로지스틱 회귀 분석에서는 사기, 비사기 트랜잭션을 상당히 정확하게 구분합니다.


**학습 곡선**
- training score와 cross validation score 사이의 차이가 클수록 모형이 과적합(고분산)할 가능성이 높다. : 오버피팅
- 교육 및 교차 검증 세트 모두에서 점수가 낮은 경우 이는 모델이 적합하지 않음을 나타낸다(높은 편향). : 언더피팅
- 로지스틱 회귀 분석 분류기는 교육 및 교차 검증 세트 모두에서 가장 높은 점수를 표시한다.

<br>

- 데이터셋


```python
from sklearn.model_selection import train_test_split

# 독립변수, 종속변수 분리
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# 학습, 시험 데이터셋 분리 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DataFrame -> array
X_train = X_train.values 
X_test = X_test.values 
y_train = y_train.values 
y_test = y_test.values 
```

- 모델 학습


```python
# 모델 정의 
classifiers = {
    'LogisticRegression': LogisticRegression(),
    'KNearest': KNeighborsClassifier(),
    'Support Vector Classifier': SVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier()
}

# 모델 학습 및 정확도 평가 
from sklearn.model_selection import cross_val_score

for key, classifier in classifiers.items():
  classifier.fit(X_train, y_train)
  training_score = cross_val_score(classifier, X_train, y_train, cv=5) 
  print('Classifiers: ', classifier.__class__.__name__, 'Has a training score of', round(training_score.mean(), 2) * 100, "% accuracy score")
```

    Classifiers:  LogisticRegression Has a training score of 94.0 % accuracy score
    Classifiers:  KNeighborsClassifier Has a training score of 94.0 % accuracy score
    Classifiers:  SVC Has a training score of 94.0 % accuracy score
    Classifiers:  DecisionTreeClassifier Has a training score of 91.0 % accuracy score
    

- GridSearchCV


```python
# GridSearchCV를 통해 최적 하이퍼파라미터를 찾는다. 
from sklearn.model_selection import GridSearchCV

# Logistic Regression
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
log_reg = grid_log_reg.best_estimator_

# KNN 
knears_params = {'n_neighbors': list(range(2, 5, 1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)
knears_neighbors = grid_knears.best_estimator_

# Support Vector Classifier 
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)
svc = grid_svc.best_estimator_

# DecisionTree Classifier 
tree_params = {'criterion': ['gini', 'entropy'], 'max_depth': list(range(2, 4, 1)),
               'min_samples_leaf': list(range(5, 7, 1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)
tree_clf = grid_tree.best_estimator_ 
```


```python
log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean()*100, 2).astype(str) + '%')

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('Knears Neighbors Cross Validation Score: ', round(knears_score.mean()*100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier Cross Validation Score: ', round(svc_score.mean()*100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('Decesion Tree Classifier Cross Validation Score: ', round(tree_score.mean()*100, 2).astype(str) + '%')
```

    Logistic Regression Cross Validation Score:  94.18%
    Knears Neighbors Cross Validation Score:  94.31%
    Support Vector Classifier Cross Validation Score:  94.44%
    Decesion Tree Classifier Cross Validation Score:  91.53%
    


```python
undersample_X = df.drop('Class', axis=1)
undersample_y = df['Class']

for train_index, test_index in sss.split(undersample_X, undersample_y):
  print("Train:", train_index, 'Test:', test_index)
  undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
  undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]

undersample_Xtrain = undersample_Xtrain.values 
undersample_Xtest = undersample_Xtest.values 
undersample_ytrain = undersample_ytrain.values
undersample_ytest = undersample_ytest.values 

undersample_accuracy = []
undersample_precision = []
undersample_recall = []
undersample_f1 = []
undersample_auc = [] 

# NearMiss Technique 
# NearMiss 분포 (우리가 사용하지 않을 이 변수들의 분포가 어떤지 확인하기 위해)
X_nearmiss, y_nearmiss = NearMiss().fit_resample(undersample_X.values, undersample_y.values)
print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))

# 올바른 방법으로 cross validating 
for train, test in sss.split(undersample_Xtrain, undersample_ytrain): 
  undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), log_reg)
  undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train]) # 언더샘플링 모델 학습 
  undersample_prediction = undersample_model.predict(undersample_Xtrain[test]) # 언더샘플링 모델 평가 (validation set) 

  undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))
  undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))
  undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))
  undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))
  undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))
```

    Train: [ 56953  56954  56955 ... 284804 284805 284806] Test: [    0     1     2 ... 59039 59380 59968]
    Train: [     0      1      2 ... 284804 284805 284806] Test: [ 56953  56954  56955 ... 113922 113923 113924]
    Train: [     0      1      2 ... 284804 284805 284806] Test: [113764 113925 113926 ... 170892 170893 170894]
    Train: [     0      1      2 ... 284804 284805 284806] Test: [166690 166692 167046 ... 227845 227846 227847]
    Train: [     0      1      2 ... 227845 227846 227847] Test: [227429 227701 227848 ... 284804 284805 284806]
    NearMiss Label Distribution: Counter({0: 492, 1: 492})
    

* undersample_Xtrain, undersample_ytrain과 original_Xtrain, original_ytrain은 같은 데이터셋이고 같은 데이터셋이어야하는게 맞는데 나눠서 사용하는 이유가 궁금하다. 


```python
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    # First Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    
    # Second Estimator 
    train_sizes, train_scores, test_scores = learning_curve(
        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)
    ax2.set_xlabel('Training size (m)')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend(loc="best")
    
    # Third Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax3.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)
    ax3.set_xlabel('Training size (m)')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    ax3.legend(loc="best")
    
    # Fourth Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax4.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)
    ax4.set_xlabel('Training size (m)')
    ax4.set_ylabel('Score')
    ax4.grid(True)
    ax4.legend(loc="best")
    return plt
```


```python
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)
```




    <module 'matplotlib.pyplot' from '/usr/local/lib/python3.7/dist-packages/matplotlib/pyplot.py'>




    
![png]({{site.baseurl}}/images/output_63_1.png)
    



```python
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
# Create a DataFrame with all the scores and the classifiers names.

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,
                             method="decision_function") # decision_function: 각 샘플의 점수를 반환 (양수: label1, 음수: label0)

knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)

svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,
                             method="decision_function")

tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)
```

\* KNN, DT에서 predict_proba를 쓰지 않아도 되는지 궁금하다


```python
from sklearn.metrics import roc_auc_score

print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))
print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))
print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))
print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))
```

    Logistic Regression:  0.9829945818477864
    KNears Neighbors:  0.9412074338171303
    Support Vector Classifier:  0.9794292692512844
    Decision Tree Classifier:  0.9125796580668707
    


```python

log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)


def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)
plt.show()
```


    
![png]({{site.baseurl}}/images/output_67_0.png)
    


### 3.5 로지스틱 회귀 분석
로지스틱 회귀 분류기에 대해 더 자세히 알아본다. 


**용어** 
- True Positives: 사기 거래로 올바르게 분류
- False Positives: 사기 거래로 잘못 분류
- True Negative: 비사기 거래로 올바르게 분류
- False Negative: 비사기 거래로 잘못 분류
- Precision(정밀도): True Positives / (True Positives + False Positives) [긍정으로 예측한 것 중 real 긍정 예측]
- Recall(민감도): True Positives / (True Positives + False Negatives) [전체 real 긍정 중 real 긍정 예측]
- 정밀도는 사기 거래를 탐지하는 데 있어 모델이 얼마나 정밀(얼마나 확실한지)한지 판단한다. 반면 리콜은 우리의 모델이 감지할 수 있는 사기 사례의 양이다.
- Precision(정밀도) & Recall(민감도) Trade-off: 모델이 더 정밀할수록(선택적) 감지되는 사례가 줄어듭니다. (예) 우리 모델의 정밀도가 95%라고 가정하면, 95% 이상 정확한 사기 건수가 5건에 불과하다고 하자. 그렇다면 우리 모델이 90%의 확률로 사기로 간주하는 사례가 5건 더 있다고 하자, 정밀도를 90%로 낮추면 우리 모델이 감지할 수 있는 사례가 5개 더 많아질 것이다.


**요약**
- 그럼에도 불구하고 정밀도는 0.90에서 0.92 사이로 떨어지기 시작하지만, 우리의 정밀도는 여전히 꽤 높고 여전히 감소하는 민감도가 있다. 


```python
def logistic_roc_curve(log_fpr, log_tpr):
    plt.figure(figsize=(12,8))
    plt.title('Logistic Regression ROC Curve', fontsize=16)
    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01,1,0,1])
    
    
logistic_roc_curve(log_fpr, log_tpr)
plt.show()
```


    
![png]({{site.baseurl}}/images/output_69_0.png)
    



```python
from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)
```


```python
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
y_pred = log_reg.predict(X_train)

print('---' * 45)
print('Overfitting: \n')
print('Recall Score: {:.2f}'.format(recall_score(y_train, y_pred)))
print('Precision Score: {:.2f}'.format(precision_score(y_train, y_pred)))
print('F1 Score: {:.2f}'.format(f1_score(y_train, y_pred)))
print('Accuracy Score: {:.2f}'.format(accuracy_score(y_train, y_pred)))
print('---' * 45)

print('---' * 45)
print('How it should be:\n')
print("Accuracy Score: {:.2f}".format(np.mean(undersample_accuracy)))
print("Precision Score: {:.2f}".format(np.mean(undersample_precision)))
print("Recall Score: {:.2f}".format(np.mean(undersample_recall)))
print("F1 Score: {:.2f}".format(np.mean(undersample_f1)))
print('---' * 45)
```

    ---------------------------------------------------------------------------------------------------------------------------------------
    Overfitting: 
    
    Recall Score: 0.93
    Precision Score: 0.79
    F1 Score: 0.85
    Accuracy Score: 0.85
    ---------------------------------------------------------------------------------------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------
    How it should be:
    
    Accuracy Score: 0.81
    Precision Score: 0.00
    Recall Score: 0.17
    F1 Score: 0.00
    ---------------------------------------------------------------------------------------------------------------------------------------
    


```python
undersample_y_score = log_reg.decision_function(original_Xtest)
```


```python
from sklearn.metrics import average_precision_score

undersample_average_precision = average_precision_score(original_ytest, undersample_y_score)

print('Average precision-recall score: {0:0.2f}'.format(undersample_average_precision))
```

    Average precision-recall score: 0.04
    


```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(original_ytest, undersample_y_score)

plt.step(recall, precision, color='#004a93', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#48a6ff')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('UnderSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(
          undersample_average_precision), fontsize=16)
```




    Text(0.5, 1.0, 'UnderSampling Precision-Recall curve: \n Average Precision-Recall Score =0.04')




    
![png]({{site.baseurl}}/images/output_74_1.png)
    


## 3.6 SMOTE를 사용한 오버샘플링(Over Smpling)

**SMOTE (Over-Sampling)**  
<!-- <img src="https://raw.githubusercontent.com/rikunert/SMOTE_visualisation/master/SMOTE_R_visualisation_3.png" width=800>    -->
SMOTE는 Synthetic Minority Over-sampling 기법이다. 랜덤 언더샘플링과 달리, SMOTE는 클래스의 균형을 같게 하기 위해 새로운 합성 점(synthetic points)를 만든다. 이는 "클래스 불균형 문제"를 해결하기 위한 또 다른 대안이다. 

**SMOTE 이해**
- 계층 불균형 해결: SMOTE는 소수 계층과 다수 계층 사이의 동등한 균형에 도달하기 위해 소수 계층으로부터 합성 포인트를 만든다.
- 합성점의 위치: SMOTE는 소수계급의 가장 가까운 이웃 사이의 거리를 선택하고, 이 거리들 사이에 합성점을 생성한다.
- 최종 효과: 랜덤 언더샘플링과 달리 행을 삭제할 필요가 없으므로 더 많은 정보가 유지된다.
- 정확도 || 시간 트레이드오프: SMOTE는 무작위 언더샘플링보다 정확할 가능성이 높지만, 앞서 언급한 바와 같이 행이 제거되지 않기 때문에 훈련하는 데 더 많은 시간이 걸릴 것이다.

**교차 유효성 검사 과적합 실수**  
- 교차 검증 중 과적합: 우리의 언더샘플 분석에서 범한 일반적인 실수가 있다. 데이터를 언더샘플링하거나 오버샘플링하려는 경우 교차 검증 전에 이 작업을 수행해서는 안된다. 교차 검증을 구현하기 전에 검증 세트에 직접 영향을 미쳐 "데이터 누출" 문제가 발생하기 때문이다. 다음 섹션에서는 놀라운 정밀도와 리콜 점수를 확인할 수 있지만 실제로는 우리의 데이터가 과적합되었다!



- 틀린 방법   

![image.png]({{site.baseurl}}/images/01.credit_1.png)  

앞에서 언급했듯이, 우리의 경우 소수 클래스("사기")를 얻고 교차 검증 전에 합성 포인트를 생성하면 교차 검증 프로세스의 "검증 세트"에 일정한 영향을 미친다. 교차 검증이 어떻게 작동하는지 기억해야한다. 데이터를 5개의 배치로 분할한다고 가정하면, 데이터 세트의 4/5가 교육 세트이고 1/5이 검증 세트인데, 테스트 세트를 만지면 안 된다. 


따라서 다음과 같이 이전이 아닌 교차 검증 중에 합성 데이터 지점을 생성해야 한다.

- 맞는 방법   

![image.png]({{site.baseurl}}/images/01.credit_2.png)  

위에서 보듯이 SMOTE는 교차 검증 프로세스 이전이 아니라 교차 검증 중에 발생해야한다. 합성 데이터는 검증 세트에 영향을 주지 않고 교육 세트에 대해서만 생성된다.

**[References]**  
-DEALING WITH IMBALANCED DATA: UNDERSAMPLING, OVERSAMPLING AND PROPER CROSS-VALIDATION  
-SMOTE explained for noobs  
-Machine Learning - Over-& Undersampling - Python/ Scikit/ Scikit-Imblearn


```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV

print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))
print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))

# 리스트에 점수를 추가하고 평균을 구한다.
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

# Classifier with optimal parameters
# log_reg_sm = grid_log_reg.best_estimator_
log_reg_sm = LogisticRegression()




rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)


# Implementing SMOTE Technique 
# Cross Validating the right way
# Parameters
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
for train, test in sss.split(original_Xtrain, original_ytrain):
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..
    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
    best_est = rand_log_reg.best_estimator_
    prediction = best_est.predict(original_Xtrain[test])
    
    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))
    precision_lst.append(precision_score(original_ytrain[test], prediction))
    recall_lst.append(recall_score(original_ytrain[test], prediction))
    f1_lst.append(f1_score(original_ytrain[test], prediction))
    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))
    
print('---' * 45)
print('')
print("accuracy: {}".format(np.mean(accuracy_lst)))
print("precision: {}".format(np.mean(precision_lst)))
print("recall: {}".format(np.mean(recall_lst)))
print("f1: {}".format(np.mean(f1_lst)))
print('---' * 45)
```

    Length of X (train): 227846 | Length of y (train): 227846
    Length of X (test): 56961 | Length of y (test): 56961
    ---------------------------------------------------------------------------------------------------------------------------------------
    
    accuracy: 0.9417990800284043
    precision: 0.06121501623610478
    recall: 0.9162934112301201
    f1: 0.113010549626039
    ---------------------------------------------------------------------------------------------------------------------------------------
    


```python
labels = ['No Fraud', 'Fraud']
smote_prediction = best_est.predict(original_Xtest)
print(classification_report(original_ytest, smote_prediction, target_names=labels))
```

                  precision    recall  f1-score   support
    
        No Fraud       1.00      0.99      0.99     56863
           Fraud       0.10      0.86      0.17        98
    
        accuracy                           0.99     56961
       macro avg       0.55      0.92      0.58     56961
    weighted avg       1.00      0.99      0.99     56961
    
    


```python
y_score = best_est.decision_function(original_Xtest)
```


```python
average_precision = average_precision_score(original_ytest, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
```

    Average precision-recall score: 0.70
    


```python
fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(original_ytest, y_score)

plt.step(recall, precision, color='r', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#F59B00')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('OverSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(
          average_precision), fontsize=16)
```




    Text(0.5, 1.0, 'OverSampling Precision-Recall curve: \n Average Precision-Recall Score =0.70')




    
![png]({{site.baseurl}}/images/output_83_1.png)
    



```python
# SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE('minority', random_state=42)
# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)


# This will be the data were we are going to 
Xsm_train, ysm_train = sm.fit_resample(original_Xtrain, original_ytrain)
```


```python
# We Improve the score by 2% points approximately 
# Implement GridSearchCV and the other models.

# Logistic Regression
t0 = time.time()
log_reg_sm = grid_log_reg.best_estimator_
log_reg_sm.fit(Xsm_train, ysm_train)
t1 = time.time()
print("Fitting oversample data took :{} sec".format(t1 - t0))
```

    Fitting oversample data took :14.712885856628418 sec
    

# 4.테스팅


## 4.1 로지스틱 회귀 분석을 사용한 검정(Testing)
**혼동행렬 Confusion Matrix**  
- Positive/Negative: 클래스(레이블)의 타입 ["No", "Yes"] 
- True/False: 모델의 분류가 맞으면 True, 틀리면 False

<br>

- True Negatives (Top-Left Square): 비사기거래 class로 올바르게 분류한 개수 


- False Negatives (Top-Right Square): 사기거래 class로 틀리게 분류한 개수 (실제로는 비사기거래 클래스)


- False Positives (Bottom-Left Square): 비사기거래 class로 틀리게 분류한 개수 (실제로는 사기거래)


- True Positives (Bottom-Right Square): 사기거래 class로 올바르게 분류한 개수 


**요약**
- 랜덤 언더 샘플링: 우리는 랜덤 언더샘플링 하위 집합에서 분류 모델의 최종 성능을 평가한다. 이 데이터는 원래 데이터 프레임의 데이터가 아니다.
- 분류 모델: 가장 우수한 성능을 보인 모델은 로지스틱 회귀 분석 및 서포트벡터머신(SVM)이었다.



```python
from sklearn.metrics import confusion_matrix

# SMOTE기법을 사용한 Logistic Regression 모델로 예측 
y_pred_log_reg = log_reg_sm.predict(X_test)

# 언더샘플링으로 학습한 다른 모델들로 예측 
y_pred_knear = knears_neighbors.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_tree = tree_clf.predict(X_test) 

log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
svc_cf = confusion_matrix(y_test, y_pred_svc)
tree_cf = confusion_matrix(y_test, y_pred_tree)

fig, ax = plt.subplots(2, 2, figsize=(22, 12))

sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper)
ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
ax[0, 0].set_xticklabels(['0(pred))', '1(pred)'], fontsize=14, rotation=0)
ax[0, 0].set_yticklabels(['0', '1'], fontsize=14, rotation=360)

sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, cmap=plt.cm.copper)
ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix", fontsize=14)
ax[0][1].set_xticklabels(['0(pred)', '1(pred)'], fontsize=14, rotation=0)
ax[0][1].set_yticklabels(['0', '1'], fontsize=14, rotation=360)

sns.heatmap(svc_cf, ax=ax[1][0], annot=True, cmap=plt.cm.copper)
ax[1][0].set_title("Suppor Vector Classifier \n Confusion Matrix", fontsize=14)
ax[1][0].set_xticklabels(['0(pred)', '1(pred)'], fontsize=14, rotation=0)
ax[1][0].set_yticklabels(['0', '1'], fontsize=14, rotation=360)

sns.heatmap(tree_cf, ax=ax[1][1], annot=True, cmap=plt.cm.copper)
ax[1][1].set_title("DecisionTree Classifier \n Confusion Matrix", fontsize=14)
ax[1][1].set_xticklabels(['0(pred)', '1(pred)'], fontsize=14, rotation=0)
ax[1][1].set_yticklabels(['0', '1'], fontsize=14, rotation=360)
```




    [Text(0, 0.5, '0'), Text(0, 1.5, '1')]




    
![png]({{site.baseurl}}/images/output_88_1.png)
    



```python
from sklearn.metrics import classification_report


print('Logistic Regression:')
print(classification_report(y_test, y_pred_log_reg))

print('KNears Neighbors:')
print(classification_report(y_test, y_pred_knear))

print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_svc))

print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_tree))
```

    Logistic Regression:
                  precision    recall  f1-score   support
    
               0       0.97      0.97      0.97        90
               1       0.97      0.97      0.97        99
    
        accuracy                           0.97       189
       macro avg       0.97      0.97      0.97       189
    weighted avg       0.97      0.97      0.97       189
    
    KNears Neighbors:
                  precision    recall  f1-score   support
    
               0       0.94      0.91      0.93        90
               1       0.92      0.95      0.94        99
    
        accuracy                           0.93       189
       macro avg       0.93      0.93      0.93       189
    weighted avg       0.93      0.93      0.93       189
    
    Support Vector Classifier:
                  precision    recall  f1-score   support
    
               0       0.97      0.94      0.96        90
               1       0.95      0.97      0.96        99
    
        accuracy                           0.96       189
       macro avg       0.96      0.96      0.96       189
    weighted avg       0.96      0.96      0.96       189
    
    Support Vector Classifier:
                  precision    recall  f1-score   support
    
               0       0.96      0.86      0.91        90
               1       0.88      0.97      0.92        99
    
        accuracy                           0.92       189
       macro avg       0.92      0.91      0.91       189
    weighted avg       0.92      0.92      0.91       189
    
    


```python
# Logistic Regression의 마지막 시험 데이터셋 점수 
from sklearn.metrics import accuracy_score 

# 언더샘플링을 사용한 Logistic Regression 모델로 에측 
y_pred = log_reg.predict(X_test) 
undersample_score = accuracy_score(y_test, y_pred)

# SMOTE 기법을 사용한 Logistic Regression 모델(rand_log_reg)로 예측 
y_pred_sm = best_est.predict(original_Xtest)
oversample_score = accuracy_score(original_ytest, y_pred_sm)

d = {'Technique': ['Random UnderSampling', 'Oversampling(SMOTE)'], 'Score': [undersample_score, oversample_score]}
final_df = pd.DataFrame(data=d)

final_df
```





  <div id="df-cf841a76-5638-41cf-a335-9c37cdd99bb2">
    <div class="colab-df-container">
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
      <th>Technique</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Random UnderSampling</td>
      <td>0.968254</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Oversampling(SMOTE)</td>
      <td>0.986096</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-cf841a76-5638-41cf-a335-9c37cdd99bb2')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-cf841a76-5638-41cf-a335-9c37cdd99bb2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cf841a76-5638-41cf-a335-9c37cdd99bb2');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## 4.2 신경망 테스팅(언더샘플링 vs 오버샘플링)
간단한 신경망 (hidden layer 1개) 을 구현하여 언더샘플링(sub_sample)과 오버샘플링(SMOTE)에서 구현한 두 로지스틱 회귀 모델 중 어떤 것이 사기거래 탐지에 더 나은 정확도를 가지고 있는지 확인한다. 

**주요 목표**   
- 우리의 주요 목표는 간단한 신경망이 무작위 언더샘플 및 오버샘플 데이터프레임에서 어떻게 행동하는지 탐색하고, 사기, 비사기거래를 정확히 예측할 수 있는지 확인하는 것이다. 
- 사기에만 집중하지 않는 이유는 다음과 같다. 본인의 카드로 물건을 구입하였는데, 은행의 알고리즘이 해당 구매를 사기라고 예측하여 차단 당하는 경우가 생길 수 있다. 따라서 사기 사건 적발에만 중점을 두어서는 안 되며, 비사기 거래를 정확히 분류하는 것도 강조해야 한다. 


**요약(keras || 랜덤 언더 샘플링)**  
- 데이터 집합: 이 테스트의 마지막 단계에서 우리는 원본 데이터의 테스트 데이터를 사용하여 최종 결과를 예측하기 위해 무작위 언더샘플링 서브셋과 오버샘플링 데이터셋(SMOTE) 을 사용한 모델을 학습한다. 
- 신경망 구조: 앞서 언급한 바와 같이, 이것은 하나의 input layer(노드의 수가 변수의 개수와 동일)과 편향 노드, 32개의 노드를 가진 hidden layer, 그리고 두 가지 가능한 결과 0 또는 1로 구성된 하나의 output layer로 구성된 간단한 모델이 될 것이다(사기:1, 비사기:0).
- 기타 특성: learning rate는 0.001이 될 것이고, 우리가 사용할 최적화 도구는 Adam Optimizer이고, 이 시나리오에서 사용되는 활성화 함수는 "ReLu"이며, 최종 출력에 대해서는 사기 또는 비사기의 확률을 제공하는 sparse categorical cross entropy를 사용할 것이다(예측은 둘 중 더 높은 확률을 가진 범주로 선택할 것이다).


```python
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

n_inputs = X_train.shape[1]

undersample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
```


```python
undersample_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 30)                930       
                                                                     
     dense_1 (Dense)             (None, 32)                992       
                                                                     
     dense_2 (Dense)             (None, 2)                 66        
                                                                     
    =================================================================
    Total params: 1,988
    Trainable params: 1,988
    Non-trainable params: 0
    _________________________________________________________________
    


```python
undersample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```


```python
undersample_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose=2)
```

    Epoch 1/20
    25/25 - 2s - loss: 0.4808 - accuracy: 0.7715 - val_loss: 0.3570 - val_accuracy: 0.8882 - 2s/epoch - 69ms/step
    Epoch 2/20
    25/25 - 0s - loss: 0.3328 - accuracy: 0.8775 - val_loss: 0.2672 - val_accuracy: 0.9474 - 129ms/epoch - 5ms/step
    Epoch 3/20
    25/25 - 0s - loss: 0.2634 - accuracy: 0.9189 - val_loss: 0.2129 - val_accuracy: 0.9539 - 130ms/epoch - 5ms/step
    Epoch 4/20
    25/25 - 0s - loss: 0.2122 - accuracy: 0.9288 - val_loss: 0.1904 - val_accuracy: 0.9605 - 140ms/epoch - 6ms/step
    Epoch 5/20
    25/25 - 0s - loss: 0.1763 - accuracy: 0.9387 - val_loss: 0.1643 - val_accuracy: 0.9605 - 116ms/epoch - 5ms/step
    Epoch 6/20
    25/25 - 0s - loss: 0.1507 - accuracy: 0.9454 - val_loss: 0.1595 - val_accuracy: 0.9539 - 113ms/epoch - 5ms/step
    Epoch 7/20
    25/25 - 0s - loss: 0.1300 - accuracy: 0.9536 - val_loss: 0.1482 - val_accuracy: 0.9474 - 107ms/epoch - 4ms/step
    Epoch 8/20
    25/25 - 0s - loss: 0.1185 - accuracy: 0.9553 - val_loss: 0.1535 - val_accuracy: 0.9408 - 75ms/epoch - 3ms/step
    Epoch 9/20
    25/25 - 0s - loss: 0.1050 - accuracy: 0.9586 - val_loss: 0.1518 - val_accuracy: 0.9474 - 79ms/epoch - 3ms/step
    Epoch 10/20
    25/25 - 0s - loss: 0.0981 - accuracy: 0.9586 - val_loss: 0.1558 - val_accuracy: 0.9408 - 77ms/epoch - 3ms/step
    Epoch 11/20
    25/25 - 0s - loss: 0.0889 - accuracy: 0.9636 - val_loss: 0.1577 - val_accuracy: 0.9408 - 73ms/epoch - 3ms/step
    Epoch 12/20
    25/25 - 0s - loss: 0.0824 - accuracy: 0.9652 - val_loss: 0.1593 - val_accuracy: 0.9408 - 81ms/epoch - 3ms/step
    Epoch 13/20
    25/25 - 0s - loss: 0.0756 - accuracy: 0.9652 - val_loss: 0.1669 - val_accuracy: 0.9408 - 97ms/epoch - 4ms/step
    Epoch 14/20
    25/25 - 0s - loss: 0.0699 - accuracy: 0.9719 - val_loss: 0.1735 - val_accuracy: 0.9408 - 87ms/epoch - 3ms/step
    Epoch 15/20
    25/25 - 0s - loss: 0.0658 - accuracy: 0.9768 - val_loss: 0.1755 - val_accuracy: 0.9408 - 105ms/epoch - 4ms/step
    Epoch 16/20
    25/25 - 0s - loss: 0.0619 - accuracy: 0.9801 - val_loss: 0.1788 - val_accuracy: 0.9408 - 91ms/epoch - 4ms/step
    Epoch 17/20
    25/25 - 0s - loss: 0.0573 - accuracy: 0.9801 - val_loss: 0.1849 - val_accuracy: 0.9408 - 83ms/epoch - 3ms/step
    Epoch 18/20
    25/25 - 0s - loss: 0.0536 - accuracy: 0.9834 - val_loss: 0.1914 - val_accuracy: 0.9408 - 77ms/epoch - 3ms/step
    Epoch 19/20
    25/25 - 0s - loss: 0.0510 - accuracy: 0.9851 - val_loss: 0.1907 - val_accuracy: 0.9408 - 84ms/epoch - 3ms/step
    Epoch 20/20
    25/25 - 0s - loss: 0.0474 - accuracy: 0.9884 - val_loss: 0.1984 - val_accuracy: 0.9408 - 76ms/epoch - 3ms/step
    




    <keras.callbacks.History at 0x7fd234760090>




```python
undersample_predictions = undersample_model.predict(original_Xtest, batch_size=200, verbose=0)
undersample_fraud_predictions = undersample_predictions.argmax(axis=-1)
```


```python
import itertools

# 혼동 행렬을 생성한다. 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```


```python
undersample_cm = confusion_matrix(original_ytest, undersample_fraud_predictions)
actual_cm = confusion_matrix(original_ytest, original_ytest)
labels = ['No Fraud', 'Fraud']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(undersample_cm, labels, title="Random UnderSample \n Confusion Matrix", cmap=plt.cm.Reds)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)
```

    Confusion matrix, without normalization
    [[52444  4419]
     [    3    95]]
    Confusion matrix, without normalization
    [[56863     0]
     [    0    98]]
    


    
![png]({{site.baseurl}}/images/output_98_1.png)
    


**Keras || OverSampling (SMOTE)**


```python
n_inputs = Xsm_train.shape[1]

oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
```


```python
oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```


```python
oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)
```

    Epoch 1/20
    1214/1214 - 6s - loss: 0.0780 - accuracy: 0.9727 - val_loss: 0.0387 - val_accuracy: 0.9837 - 6s/epoch - 5ms/step
    Epoch 2/20
    1214/1214 - 4s - loss: 0.0191 - accuracy: 0.9944 - val_loss: 0.0111 - val_accuracy: 0.9989 - 4s/epoch - 3ms/step
    Epoch 3/20
    1214/1214 - 3s - loss: 0.0110 - accuracy: 0.9976 - val_loss: 0.0057 - val_accuracy: 0.9996 - 3s/epoch - 2ms/step
    Epoch 4/20
    1214/1214 - 3s - loss: 0.0074 - accuracy: 0.9984 - val_loss: 0.0031 - val_accuracy: 0.9999 - 3s/epoch - 2ms/step
    Epoch 5/20
    1214/1214 - 3s - loss: 0.0056 - accuracy: 0.9989 - val_loss: 0.0064 - val_accuracy: 0.9999 - 3s/epoch - 2ms/step
    Epoch 6/20
    1214/1214 - 3s - loss: 0.0044 - accuracy: 0.9991 - val_loss: 0.0052 - val_accuracy: 0.9999 - 3s/epoch - 2ms/step
    Epoch 7/20
    1214/1214 - 4s - loss: 0.0040 - accuracy: 0.9992 - val_loss: 0.0120 - val_accuracy: 0.9997 - 4s/epoch - 3ms/step
    Epoch 8/20
    1214/1214 - 4s - loss: 0.0031 - accuracy: 0.9994 - val_loss: 0.0030 - val_accuracy: 0.9998 - 4s/epoch - 3ms/step
    Epoch 9/20
    1214/1214 - 3s - loss: 0.0030 - accuracy: 0.9995 - val_loss: 0.0014 - val_accuracy: 1.0000 - 3s/epoch - 2ms/step
    Epoch 10/20
    1214/1214 - 3s - loss: 0.0024 - accuracy: 0.9995 - val_loss: 0.0014 - val_accuracy: 1.0000 - 3s/epoch - 2ms/step
    Epoch 11/20
    1214/1214 - 3s - loss: 0.0021 - accuracy: 0.9996 - val_loss: 0.0014 - val_accuracy: 1.0000 - 3s/epoch - 2ms/step
    Epoch 12/20
    1214/1214 - 3s - loss: 0.0020 - accuracy: 0.9996 - val_loss: 7.3873e-04 - val_accuracy: 1.0000 - 3s/epoch - 2ms/step
    Epoch 13/20
    1214/1214 - 3s - loss: 0.0017 - accuracy: 0.9997 - val_loss: 0.0012 - val_accuracy: 1.0000 - 3s/epoch - 2ms/step
    Epoch 14/20
    1214/1214 - 3s - loss: 0.0016 - accuracy: 0.9997 - val_loss: 6.8631e-04 - val_accuracy: 0.9999 - 3s/epoch - 2ms/step
    Epoch 15/20
    1214/1214 - 3s - loss: 0.0015 - accuracy: 0.9997 - val_loss: 8.9564e-04 - val_accuracy: 1.0000 - 3s/epoch - 2ms/step
    Epoch 16/20
    1214/1214 - 3s - loss: 0.0013 - accuracy: 0.9997 - val_loss: 0.0025 - val_accuracy: 0.9997 - 3s/epoch - 2ms/step
    Epoch 17/20
    1214/1214 - 3s - loss: 0.0016 - accuracy: 0.9997 - val_loss: 4.8170e-04 - val_accuracy: 1.0000 - 3s/epoch - 2ms/step
    Epoch 18/20
    1214/1214 - 3s - loss: 0.0011 - accuracy: 0.9997 - val_loss: 0.0022 - val_accuracy: 0.9994 - 3s/epoch - 2ms/step
    Epoch 19/20
    1214/1214 - 3s - loss: 9.5084e-04 - accuracy: 0.9998 - val_loss: 3.5280e-04 - val_accuracy: 1.0000 - 3s/epoch - 2ms/step
    Epoch 20/20
    1214/1214 - 3s - loss: 0.0013 - accuracy: 0.9997 - val_loss: 5.2458e-04 - val_accuracy: 1.0000 - 3s/epoch - 2ms/step
    




    <keras.callbacks.History at 0x7fd234020750>




```python
oversample_predictions = oversample_model.predict(original_Xtest, batch_size=200, verbose=0)
oversample_fraud_predictions = oversample_predictions.argmax(axis=-1)
```


```python
oversample_smote = confusion_matrix(original_ytest, oversample_fraud_predictions)
actual_cm = confusion_matrix(original_ytest, original_ytest)
labels = ['No Fraud', 'Fraud']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(oversample_smote, labels, title="OverSample (SMOTE) \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)
```

    Confusion matrix, without normalization
    [[56838    25]
     [   27    71]]
    Confusion matrix, without normalization
    [[56863     0]
     [    0    98]]
    


    
![png]({{site.baseurl}}/images/output_104_1.png)
    


**결론**  
- 불균형 데이터 세트에 SMOTE를 구현하면 레이블의 불균형을 해결할 수 있었다.
- 그럼에도 불구하고, 여전히 때때로 오버샘플 데이터셋의 신경망이 언더샘플 데이터셋을 사용하는 모델보다 덜 정확한 부정 거래를 예측한다는 것을 말해야 한다.
- 그러나 이상치 제거는 오버샘플 데이터셋이 아닌 랜덤 언더샘플 데이터셋에서만 구현되었다는 것을 기억해야한다.
- 또한, 우리의 언더샘플 데이터셋에서 우리 모델은 비사기 거래를 사기 거래로 잘못 분류하는 사례가 많았다. 많은 비사기 거래 대해 올바르게 감지할 수 없다고 판단할 수 있다.   
- 우리의 모델이 어떠한 거래를 사기 거래로 분류했다는 이유로 정기 구매를 하던 사람들이 그들의 카드를 차단당했다고 상상해 보자. 이것은 금융 기관에 큰 불이익이 될 것이며, 고객 불만과 고객 불만이 늘어날 것이다.
- 이 분석의 다음 단계는 오버샘플 데이터 세트에서 이상치를 제거하고 테스트 세트의 정확도가 향상되는지 확인하는 것이다.


**참고:** 마지막으로 두 가지 유형의 데이터 프레임에 데이터 셔플을 구현했기 때문에 예측과 정확성이 변경될 수 있다. 가장 중요한 것은 우리의 모델이 사기 및 사기 거래를 올바르게 분류할 수 있는지 확인하는 것이다.
