---  
layout: post  
title: python 기초 || 3.2 흐름제어 - 반복문 for, while
image: 
tags:  
categories: python
---


# 3-2 흐름제어: 반복문 for, while

# (1) While 문

- **예제1**


```python
v1 = 1
while v1 < 11: 
    print("v1 is: ", v1)
    v1 += 1 
```

    v1 is:  1
    v1 is:  2
    v1 is:  3
    v1 is:  4
    v1 is:  5
    v1 is:  6
    v1 is:  7
    v1 is:  8
    v1 is:  9
    v1 is:  10
    

**while문**은 **조건이 참일 동안** 계속 **반복**한다. 이 예제에서는 11보다 작을 때, 즉 10까지 반복한다. 

- **예제2**


```python
sum1 = 0 
cnt1 = 1
```


```python
while cnt1 <= 100: 
    sum1 += cnt1
    cnt1 += 1 
print(sum1)
```

    5050
    

**cnt1**이 **100 이하**일 동안 반복문을 실행하게 된다. **sum1**에 **1**씩 증가하는 **cnt1**을 계속 더해줌으로써 **1부터 100까지의 합**을 구하는 코드이다. 

- **참고: 같은 결과를 낼 수 있는 sum**


```python
print(sum(range(1, 101)))
print(sum(range(1, 101, 2)))
```

    5050
    2500
    

더 간단한 방법으로 합을 구해줄 수 있다. 1부터 100까지 홀수의 합만을 구하려면 **range(1, 101, 2)** 를 사용하면 된다. 

# (2) for 문

* 시퀀스 (순서가 있는) 자료형 반복 : 문자열, 리스트, 튜플, 집합, 사전 

* iterable 리턴 함수 : range. reversed, enumerate, filter, map, zip 

<BR>

- **예제 1 - range()**


```python
for v3 in range(1, 11):
    print('v3 is: ', v3)
```

    v3 is:  1
    v3 is:  2
    v3 is:  3
    v3 is:  4
    v3 is:  5
    v3 is:  6
    v3 is:  7
    v3 is:  8
    v3 is:  9
    v3 is:  10
    

시퀀스 자료형을 반복한다. 위와 같은 경우에서는 1부터 10까지 차례대로 v3에 들어가는 것을 반복하며 출력해 준다.

- **예제 2 - 리스트**


```python
names = ['kim', 'park', 'cho', 'choi', 'yoo']
```


```python
for name in names:
    print('Your name is ', name)
```

    Your name is  kim
    Your name is  park
    Your name is  cho
    Your name is  choi
    Your name is  yoo
    

**리스트 자료형**을 넣어주어도 반복이 가능하다.

- **예제 3 - 문자열**


```python
word = 'dreams'
```


```python
for s in word: 
    print("Word: ", s)
```

    Word:  d
    Word:  r
    Word:  e
    Word:  a
    Word:  m
    Word:  s
    

문자열도 반복이 가능하다. 

- **예제 4 - 딕셔너리**


```python
my_info = {
    'name': 'Silver',
    'age': 33,
    'city': 'Seoul'
}
```

- 기본: key


```python
for key in my_info:
    print(key)
```

    name
    age
    city
    

딕셔너리 자체를 반복문에 넣으면 기본적으로 키가 반복된다. 

- values


```python
for val in my_info.values():
    print(val)
```

    Silver
    33
    Seoul
    

- keys


```python
for key in my_info.keys():
    print(key)
```

    name
    age
    city
    

- items


```python
for key, val in my_info.items():
    print(key, '|', val)
```

    name | Silver
    age | 33
    city | Seoul
    

- **예제 5 - 반복문 + 조건문**


```python
name = 'KennRY'
name2 = ""
```

KennRY 에서 대문자는 소문자로, 소문자는 대문자로 바꾸고, 최종적으로 name2에 바뀐 문자열을 저장하는 코드를 생성한다. 


```python
for n in name:
    if n.isupper():
        print(n.lower())
        name2 += n.lower()
    else: 
        print(n.upper())
        name2 += n.upper()
```

    k
    E
    N
    N
    r
    y
    


```python
print(name2)
```

    kENNry
    

문자열을 반복하여, 글자가 대문자면 소문자로 바꾸고 name2에 글자를 더해준다. 그 반대의 경우도 동일하게 해준다. 원하는 결과가 나오는 것을 볼 수 있다. 

- **예제 6 - break**


```python
numbers = [14, 3, 4, 7, 10, 24, 17, 2, 33, 35, 36, 38]
numbers2 = [14, 3, 4, 7, 10, 24, 17, 2, 37, 35, 36, 38]
```


```python
for num in numbers: 
    if num == 33:
        print("found 33 !!!")
        break
    else:
        print("not found 33 !!!")
```

    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    found 33 !!!
    

33을 발견하면 **break**를 사용하여 **반복문을 탈출**하는 코드이다. 실제로 33 차례가 왔을 때, found : 33 ! 이 출력되고, 그 이후로는 아무것도 출력되지 않는 것을 볼 수 있다. 

 - **예제 7 - for-else 구문**
- 모든값 정상 순회 -> else 블럭 수행


```python
for num in numbers2:
    if num == 33:
        print('found 33 !!!')
        break
    else:
        print('not found 33 !!!')
else:
    print("Not found 33.........")
```

    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    not found 33 !!!
    Not found 33.........
    

**break**문 등으로 인하여 **반복문이 정상적으로 모든 값을 순회하지 못할 경우 else블럭은 수행되지 않는다.** 예제7과 같은 경우에는 33이 리스트에 존재하지 않기 때문에 **정상적으로 모든 값을 순회하여 else블럭이 수행**되었다. 

- **예제 8 - continue**


```python
l1 = ['1', 2, 5, True, 4.3, complex(4)]
```


```python
for i in l1:
    if type(i) is float:
        continue
    print("type: ", type(i))
```

    type:  <class 'str'>
    type:  <class 'int'>
    type:  <class 'int'>
    type:  <class 'bool'>
    type:  <class 'complex'>
    

**continue**를 만나면 그 다음 코드를 실행하지 않고 곧바로 for문의 첫번째로 되돌아 가 반복을 그대로 수행한다. 예제8에서는 float타입이면 continue를 실행한다. float 타입인 4.3을 중간에 넣어놓았더니 그 후의 print문은 실행되지 않고 complex타입은 정상적으로 출력된 것을 볼 수 있다.

- **예제 9 - reversed 함수**


```python
name = 'Niceman'
print(reversed(name))
print(list(reversed(name)))
print(tuple(reversed(name)))
```

    <reversed object at 0x000001F1A745AF60>
    ['n', 'a', 'm', 'e', 'c', 'i', 'N']
    ('n', 'a', 'm', 'e', 'c', 'i', 'N')
    

**reversed** 함수는 그대로 출력해서 보기 위해 **list**나 **tuple객체**로 변환해주어야 한다. 


```python
for n in reversed(name): 
    print(n)
```

    n
    a
    m
    e
    c
    i
    N
    

하지만 **iterable**을 반환해주는 함수이기 때문에 for문에서는 곧바로 사용할 수 있다. 
