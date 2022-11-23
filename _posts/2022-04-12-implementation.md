---  
layout: post  
title: Algorithm || 구현 Implementation
image: 02.implementation_1.png
tags:  
categories: algorithm
---

# 구현 Implementation   
\- 참고 교재: 이것이 코딩테스트다 

![image.png]({{site.baseurl}}/images/02.implementation_1.png)  
구현 유형의 문제는 **'풀이를 떠올리는 것은 쉽지만 소스코드로 옮기기 어려운 문제'**를 의미한다. 즉, 위의 사진과 같이 머릿속에 있는 알고리즘을 소스코드로 바꾸는 과정이다. 

**알고리즘 대회에서 구현 유형의 문제**  
- 풀이를 떠올리는 것은 쉽지만 소스코드로 옮기기 어려운 문제   

**구현 유형의 예시**
- 알고리즘은 간단하지만 코드가 지나치게 길어지는 문제 
- 실수 연산을 다루고, 특정 소수점 자리까지 출력해야하는 문제
- 문자열을 특정한 기준에 따라 끊어 처리해야 하는 문제 
- 적절한 라이브러리를 찾아 사용해야 하는 문제 

**알고리즘 문제에서의 2차원 공간 = `행렬(Matrix)`**


```python
for i in range(5):
    for j in range(5):
        print(f'({i}, {j})', end=' ')
    print()
```

    (0, 0) (0, 1) (0, 2) (0, 3) (0, 4) 
    (1, 0) (1, 1) (1, 2) (1, 3) (1, 4) 
    (2, 0) (2, 1) (2, 2) (2, 3) (2, 4) 
    (3, 0) (3, 1) (3, 2) (3, 3) (3, 4) 
    (4, 0) (4, 1) (4, 2) (4, 3) (4, 4) 
    

**`시뮬레이션` 및 `완전 탐색` 문제에서는 2차원 공간에서의 `방향 벡터`가 자주 활용된다.**


```python
# 동, 북, 서, 남
dx = [0, -1, 0, 1]
dy = [1, 0, -1, 0]

# 현재 위치 
x, y = 2, 2
for i in range(4): 
    # 다음 위치 
    nx, ny = x + dx[i], y + dy[i]
    print(nx, ny)
```

    2 3
    1 2
    2 1
    3 2
    

# 문제 1. 상하좌우
>여행가 A는 N × N 크기의 정사각형 공간 위에 서 있습니다. 이 공간은 1 × 1 크기의 정사각형으로 나누어져 있습니다. 가장 왼쪽 위 좌표는 (1, 1)이며, 가장 오른쪽 아래 좌표는 (N, N)에 해당합니다. 여행가 A는 상, 하, 좌, 우 방향으로 이동할 수 있으며, 시작 좌표는 항상 (1, 1)입니다. 우리 앞에는 여행가 A가 이동할 계획이 적힌 계획서가 놓여 있습니다.
계획서에는 하나의 줄에 띄어쓰기를 기준으로 하여 L, R, U, D 중 하나의 문자가 반복적으로 적혀 있습니다. 각 문자의 의미는 다음과 같습니다.  
<br>
`L: 왼쪽으로 한 칸 이동`  
`R: 오른쪽으로 한 칸 이동`  
`U: 위로 한 칸 이동`  
`D: 아래로 한 칸 이동`  
<br>
이때 여행가 A가 N × N 크기의 정사각형 공간을 벗어나는 움직임은 무시됩니다. 예를 들어 (1, 1)의 위치에서 L 혹은 U를 만나면 무시됩니다. 다음은 N = 5인 지도와 계획서입니다.
![image.png]({{site.baseurl}}/images/02.implementation_4.png)  
<br>
출력: 첫째 줄에 여행가 A가 최종적으로 도착할 지점의 좌표 (X, Y)를 공백을 기준으로 구분하여 출력


```python
import pands as pd
warnings.filterwarnings(action='ignore')

str(x).zfill()

'http://ag-su.gihtub.io'

N = int(input())
plans = input().split()

dic_move = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}
# 주석
x, y = 1, 1

for plan in plans:
    dx, dy = dic_move[plan]
    nx, ny = x + dx, y + dy
    
    if (nx < 1) or (nx > N) or (ny < 1) or (ny > N):
        continue 
    
    x, y = nx, ny 
    
print(x, y)
```

     5
     R R R U D D
    

    3 4
    

# 문제2. 시각
>정수 N이 입력되면 00시 00분 00초부터 N시 59분 59초까지의 모든 시각 중에서 3이 하나라도 포함되는 모든 경우의 수를 구하는 프로그램을 작성하세요. 예를 들어 1을 입력했을 때 다음은 3이 하나라도 포함되어 있으므로 세어야 하는 시각입니다.  
    `- 00시 00분 03초`  
    `- 00시 13분 30초`  
반면에 다음은 3이 하나도 포함되어 있지 않으므로 세면 안 되는 시각입니다.  
    `- 00시 02분 55초`  
    `- 01시 27분 45초`  


```python
n = int(input())
cnt = 0 
for h in range(n+1): 
    for m in range(60): 
        for s in range(60): 
            if '3' in str(h) + str(m) + str(s): 
                cnt += 1 
                
print(cnt)
```

     5
    

    11475
    

# 문제 3. 왕실의 나이트
>행복 왕국의 왕실 정원은 체스판과 같은 8 × 8 좌표 평면입니다. 왕실 정원의 특정한 한 칸에 나이트가 서 있습니다. 나이트는 매우 충성스러운 신하로서 매일 무술을 연마합니다.  
나이트는 말을 타고 있기 때문에 이동을 할 때는 L자 형태로만 이동할 수 있으며 정원 밖으로는 나갈 수 없습니다.  
나이트는 특정 위치에서 다음과 같은 2가지 경우로 이동할 수 있습니다.  
`1. 수평으로 두 칸 이동한 뒤에 수직으로 한 칸 이동하기`  
`2. 수직으로 두 칸 이동한 뒤에 수평으로 한 칸 이동하기`  
![image.png]({{site.baseurl}}/images/02.implementation_2.png)  
이처럼 8 × 8 좌표 평면상에서 나이트의 위치가 주어졌을 때 나이트가 이동할 수 있는 경우의 수를 출력하는 프로그램을 작성하세요. 왕실의 정원에서 행 위치를 표현할 때는 1부터 8로 표현하며, 열 위치를 표현할 때는 a부터 h로 표현합니다.  
c2에 있을 때 이동할 수 있는 경우의 수는 6가지입니다.  
![image.png]({{site.baseurl}}/images/02.implementation_3.png)  
이처럼 8 × 8 좌표 평면상에서 나이트의 위치가 주어졌을 때 나이트가 이동할 수 있는 경우의 수를 출력하는 프로그램을 작성하세요. 왕실의 정원에서 행 위치를 표현할 때는 1부터 8로 표현하며, 열 위치를 표현할 때는 a부터 h로 표현합니다.  
a1에 있을 때 이동할 수 있는 경우의 수는 2가지입니다.


```python
start = input()

x, y = ord(start[0])-96, int(start[1])

lst_move = [(-2, 1), (-2, -1), (2, 1), (2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]

cnt = 0 
for move in lst_move: 
    dx, dy = move[0], move[1]
    nx, ny = x+dx, y+dy 
    
    if (nx >= 1) and (nx <= 8) and (ny >= 1) and (ny <= 8): 
        cnt += 1 
        
print(cnt)
```

     a1
    

    2
    

# 문제 4. 문자열 재정렬
>알파벳 대문자와 숫자(0 ~ 9)로만 구성된 문자열이 입력으로 주어집니다. 이때 모든 알파벳을 오름차순으로 정렬하여 이어서 출력한 뒤에, 그 뒤에 모든 숫자를 더한 값을 이어서 출력합니다.  
예를 들어 K1KA5CB7이라는 값이 들어오면 ABCKK13을 출력합니다.


```python
str1 = input()
lst_num = []
lst_alpha = []
sum1 = 0 
for s in str1: 
    if s.isalpha(): 
        lst_alpha.append(s)
    else: 
        sum1 += int(s)
        
result = ''.join(sorted(lst_alpha)) + str(sum1)

print(result)
```

     K1KA5CB7
    

    ABCKK13
    
