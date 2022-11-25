---  
layout: post  
title: python 기초 || 2.2 datatype - 리스트, 튜플, 딕셔너리, 집합
image: 
tags:  
categories: python
---


# 2-2 Data Type : 리스트, 튜플, 딕셔너리, 집합

## (1) 리스트 (list)
- 순서 O, 중복 O, 삭제 O 

### - 선언


```python
a = []
b = list()
c = [1, 2, 3, 4]
d = [10, 100, 'Pen', 'Banana', 'Orange']
e = [10, 100, ['Pen', 'Banana', 'Orange']]
```

### - 인덱싱


```python
print(d[3])
print(d[-2])
print(d[0] + d[1])
print(e[2][1])
print(e[-1][-2])
```

    Banana
    Banana
    110
    Banana
    Banana
    

### - 슬라이싱


```python
print(d[0:3])
print(e[2][1:3])
```

    [10, 100, 'Pen']
    ['Banana', 'Orange']
    

### - 연산


```python
print(c + d)
print(c * 3)
print(str(c[0]) + 'hi')
```

    [1, 2, 3, 4, 10, 100, 'Pen', 'Banana', 'Orange']
    [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    1hi
    

### - 리스트 수정, 삭제 


```python
c[0] = 77
print(c)
```

    [77, 2, 3, 4]
    

0번째 인덱스에 값 넣어주기 


```python
c[1:2] = [100, 1000, 10000]  # 슬라이싱: 원소가 들어감 
print(c)
```

    [77, 100, 1000, 10000, 3, 4]
    

슬라이싱을 이용하여 여러 값을 넣어줄 수 있다.


```python
c[1] = ['a', 'b', 'c']  # 인덱싱: 리스트 자체가 들어감
print(c)
```

    [77, ['a', 'b', 'c'], 1000, 10000, 3, 4]
    

리스트 자체를 넣어주려면 인덱싱을 사용하면 된다. 


```python
del c[1]
print(c)
```

    [77, 1000, 10000, 3, 4]
    

del을 이용하여 리스트의 값을 삭제할 수 있다. 




```python
del c[-1]
print(c)
```

    [77, 1000, 10000, 3]
    

<br>

### - 리스트 관련 함수 

- append


```python
y = [5, 2, 3, 1, 4]
print(y)
y.append(6)
print(y)
```

    [5, 2, 3, 1, 4]
    [5, 2, 3, 1, 4, 6]
    

리스트의 가장 끝에 값을 추가해 준다. 

- sort


```python
y.sort()
print(y)
```

    [1, 2, 3, 4, 5, 6]
    

리스트를 정렬해 준다. 

- reverse


```python
y.reverse()
print(y)
```

    [1, 2, 3, 4, 5, 6]
    

리스트를 뒤집어 준다. 

- insert


```python
y.insert(2, 7)
print(y)
```

    [1, 2, 7, 3, 4, 5, 6]
    

append와는 다르게 원하는 인덱스에 값을 추가해줄 수 있다. 

- remove


```python
y.remove(2)
y.remove(7)
print(y)
```

    [1, 3, 4, 5, 6]
    

del은 인덱스를 이용하여 지워주었는데, remove함수는 값을 넣으면 그 값을 지워준다. 

- pop


```python
y.pop()
print(y)
```

    [1, 3, 4, 5]
    

pop을 사용하면 가장 끝에 있는 값을 지워준다. LIFO(Last In First Out) 의 자료구조로 사용된다. 

- extend


```python
ex = [77, 88]
# y.append(ex) # ---> 리스트 자체를 추가 함 
y.extend(ex) # 값을 추가 함
print(y)
```

    [1, 3, 4, 5, 77, 88]
    

리스트 안에 리스트를 추가하는 것이 아니라, 리스트 안에 여러 값을 한 번에 넣어주고 싶을 때 extend함수를 사용한다. 


<br>

### - 리스트 삭제 3종류 비교하기 
- del(): 인덱스 값 
- remove(): 원소 값 
- pop(): 가장 마지막 값 ( 예외 발생 주의 ) 

## (2) 튜플 (tuple)
- 순서 O, 중복 O, 수정 X, 삭제 X 

### - 선언


```python
a = ()
b = (1,)
c = (1, 2, 3, 4)
d = (10, 100, ('a', 'b', 'c'))
```

### - 삭제: 불가


```python
del c[2]  # 오류 발생
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-33-501ff3bb334c> in <module>
    ----> 1 del c[2]  # 오류 발생
    

    TypeError: 'tuple' object doesn't support item deletion


튜플은 수정이나 삭제를 할 수 없다. 

### - 인덱싱


```python
print(c[2])
print(c[3])
print(d[2][2])
```

    3
    4
    c
    

### - 슬라이싱


```python
print(d[2:])
print(d[2][0:2])
```

    (('a', 'b', 'c'),)
    ('a', 'b')
    

### - 연산


```python
print(c + d)
print(c * 3)
```

    (1, 2, 3, 4, 10, 100, ('a', 'b', 'c'))
    (1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4)
    

### - 튜플 관련 함수


```python
z = (5, 2, 1, 3, 1)
print(z)
print(3 in z)  # 튜플 안에 3이 있는가? 
print(z.index(5))  # 튜플 안의 값 5의 index를 반환 : 없으면 예외 발생 
print(z.count(1))  # 튜플 안의 값 1이 몇개 있는가 
```

    (5, 2, 1, 3, 1)
    True
    0
    2
    

## (3) 딕셔너리 (Dictionary)
- 순서 X, 중복 X, 수정 O, 삭제 O 
- Key, Value 형식으로 되어 있음. (Json/ MongoDB) 

### - 선언


```python
a = {'name': 'Kim', 'Phone': "010-7777-7777", 'birth': 990101}
b = {0: 'Hello Python', 1: 'Hello silver'}
c = {'arr': [1, 2, 3, 4, 5]}
```


```python
print(type(a), type(b), type(c))
```

    <class 'dict'> <class 'dict'> <class 'dict'>
    

딕셔너리 객체를 선언해 준다. 

 

### - 출력


```python
print(a['name'])
print(a.get('name'))
print(a.get('adress')) # 안전한 코딩 방식 -> 값이 존재하지 않아도 에러 나지 않음
print(c['arr'][1:3])
```

    Kim
    Kim
    None
    [2, 3]
    

딕셔너리에서 값을 가져오기 위한 2가지 방법이 있다. 1. 키에 직접 접근 , 2. get() 사용 

1번 방법이 편리하지만, 없는 키 값을 사용하면 오류가 나게 된다. 2번 방법을 사용하면 없는 키 값을 넣더라도 None을 반환해주기 때문에 더 안전한 코딩이 가능하다. 

### - 추가


```python
a['adress'] = 'Seoul'
print(a)
a['rank'] = [1, 3, 4]
a['rank2'] = (1, 2, 3)
print(a)
```

    {'name': 'Kim', 'Phone': '010-7777-7777', 'birth': 990101, 'adress': 'Seoul'}
    {'name': 'Kim', 'Phone': '010-7777-7777', 'birth': 990101, 'adress': 'Seoul', 'rank': [1, 3, 4], 'rank2': (1, 2, 3)}
    

key와 value를 직접 적어주면 된다. 리스트와 튜플 또한 추가가 가능하다. 

### - keys, values, items 


```python
# print(a.keys()[0])  # 리스트 인덱스로 접근 불가 
temp = list(a.keys())  # 형변환을 해주어야 함 
print(temp)
print(temp[1:3])
```

    ['name', 'Phone', 'birth', 'adress', 'rank', 'rank2']
    ['Phone', 'birth']
    

a.keys() 를 사용하면 딕셔너리 a의 key값들만 모아서 나오게 된다. 언뜻 보면 리스트처럼 생겼지만 리스트를 사용하는 것 처럼 인덱싱이나 슬라이싱을 할 수 없다. 따라서 list()를 이용하여 형변환을 해주면 키 값들이 담긴 리스트를 생성할 수 있다. 


```python
print(a.values())
print(list(a.values()))
print("-------------------------------------------------------------------------------")
print(a.items())
print(list(a.items()))
```

    dict_values(['Kim', '010-7777-7777', 990101, 'Seoul', [1, 3, 4], (1, 2, 3)])
    ['Kim', '010-7777-7777', 990101, 'Seoul', [1, 3, 4], (1, 2, 3)]
    -------------------------------------------------------------------------------
    dict_items([('name', 'Kim'), ('Phone', '010-7777-7777'), ('birth', 990101), ('adress', 'Seoul'), ('rank', [1, 3, 4]), ('rank2', (1, 2, 3))])
    [('name', 'Kim'), ('Phone', '010-7777-7777'), ('birth', 990101), ('adress', 'Seoul'), ('rank', [1, 3, 4]), ('rank2', (1, 2, 3))]
    

values()와 items()도 마찬가지다.

### - 키 존재 여부


```python
print(1 in b)
print(2 in b)
print('name' in a )
```

    True
    False
    True
    

위와 같이 딕셔너리에서 in을 이용하면 key값이 존재하는지에 대한 여부를 boolean 타입으로 반환해 준다.

## (3) 집합 (Set)
- 순서 X, 중복 X 

### - 선언


```python
a1 = {} # ----> 주의: dictionary 
a2 = set()
b = set([1, 2, 3, 4])
c = set([1, 4, 6, 6, 6])
print(type(a1), type(a2), sep='  |  ')
print(b)
print(c)
```

    <class 'dict'>  |  <class 'set'>
    {1, 2, 3, 4}
    {1, 4, 6}
    

주의할 점은 a = { } 와 같이 선언하면 딕셔너리 객체가 만들어 지므로 빈 집합을 만들 경우엔 a = set() 의 형태로 선언해 주어야 한다. 

### - 형변환
- set 자료형은 주로 형변환 하여 사용한다. (중복 값 제거 용도) 


```python
t = tuple(b)
print(t)
l = list(b)
print(l)
```

    (1, 2, 3, 4)
    [1, 2, 3, 4]
    

중복 값이 들어가지 않도록 set을 생성한 후 튜플이나 리스트 등으로 변환하여 사용하는 일이 많다고 한다.  

### - 집합연산


```python
s1 = set([1, 2, 3, 4, 5, 6])
s2 = set([4, 5, 6, 7, 8, 9])
```

연산을 수행하기 위한 집합을 선언한다. 

- **교집합: intersection / &**


```python
print(s1.intersection(s2))
print(s1 & s2)
```

    {4, 5, 6}
    {4, 5, 6}
    

- **합집합: union / |**


```python
print(s1.union(s2))
print(s1 | s2)
```

    {1, 2, 3, 4, 5, 6, 7, 8, 9}
    {1, 2, 3, 4, 5, 6, 7, 8, 9}
    

- **차집합: difference / -**


```python
print(s1.difference(s2))
print(s1 - s2)
```

    {1, 2, 3}
    {1, 2, 3}
    

- **추가 & 제거**


```python
### 추가: add()
s3 = set([7, 8, 10, 15])
s3.add(18)
s3.add(7)
print(s3)
```

    {7, 8, 10, 15, 18}
    

add함수를 이용하여 값을 추가할 수 있다.


```python
### 제거: remove()
s3.remove(15)
print(s3)
```

    {7, 8, 10, 18}
    

remove함수를 이용하여 값 제거까지 가능하다. 



