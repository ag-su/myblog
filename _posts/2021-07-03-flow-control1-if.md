---  
layout: post  
title: python 기초 || 3.1 흐름제어 - 조건문 if
image: 
tags:  
categories: python
---

# 3-1. 흐름제어: 조건문 if 

### - cf) 참(True), 거짓(False) 종류

- **참: "내용", [내용], (내용), {내용}, 1, True**
- **거짓: "", [], (), {}, 0, False**

기본적으로 여러 데이터 타입에서 내용이 채워져 있으면 참, 비어 있으면 거짓이다. 


```python
# 예 1 
if True: 
    print("Yes")
    
# 예 2 
if False:
    print('No')
    
# 예 3 
if False:
    print('No')
else:
    print('Yes')
```

    Yes
    Yes
    

이처럼 **if문**은 **True**일 때 실행되고, **False**일 때 실행되지 않는다. 이를 이용하여 위에서 배운 **참**과 **거짓**을 사용하여 **if문**을 활용한다. 


```python
city = ""

if city:
    print(">>>>>True")
else:
    print(">>>>>False")
```

    >>>>>False
    

이처럼 **비어있는 문자열**일 경우 **False**로 인식하여 **else**로 넘어가 **else**의 부분이 실행 되는 것이다. 

### - 관계연산자
- **>, >=, <, <=, ==, !=**


```python
a = 10 
b = 0 
```


```python
print(a == b)
print(a != b)
print(a > b)
print(a >= b)
print(a < b)
print(a <= b)
```

    False
    True
    True
    True
    False
    False
    

### - 논리연산자
- **and, or, not**


```python
a = 100
b = 60 
c = 15 
```


```python
print('and: ', a > b and b > c)
print('or: ', a > b or c < b)
print('not: ', not a > b)
print(not True)
print(not False)
```

    and:  True
    or:  True
    not:  False
    False
    True
    

### - 산술, 관계, 논리 연산자 적용 순서 
- **산술 > 관계 > 논리**


```python
print(5 + 10 > 0 and not 7 + 3 == 10)
```

    False
    

## (1) if문 기본 사용


```python
score1 = 90
score2 = 'A'
```


```python
if score1 >= 90 and score2 == "A":
    print('합격하셨습니다.')
else: 
    print("불합격입니다. ")
```

    합격하셨습니다.
    

score1이 90이상이면서 score2가 A이므로 (True & True = True) if문이 실행된다. 

## (2) 다중 조건문


```python
num = 82
```


```python
if num >= 90:
    print("등급 A | 점수: ", num)
elif num >= 80 :
    print("등급 B | 점수: ", num)
elif num >= 70: 
    print("등급 C | 점수: ", num)
else:
    print("꽝")
```

    등급 B | 점수:  82
    

num 이 82 이므로 등급 B가 출력된다. elif는 else if 의 줄임말이다. if가 실행되지 않으면 elif로 가게 된다. 만약 위의 경우에서 elif 대신 if를 쓴다면 ,

<br>

num 등급 B 82   
num 등급 C 82   
<br>

위와 같은 결과가 나올 것이다. 윗줄의 if문이 실행이 되든 안되든 모든 if문을 검사하기 때문이다. 이것이 elif가 필요한 이유다. 

## (3) 중첩 조건문


```python
age = 27
height = 175
```


```python
if age >= 20:
    if height >= 170: 
        print("A 지망 지원 가능 ")
    elif height >= 160:
        print("B 지망 지원 가능")
    else: 
        print(" 지원 불가")
else: 
    print("20세 이상 지원 가능 ")
```

    A 지망 지원 가능 
    

이처럼 if문 안에 if문을 사용할 수 있다. age가 20 이상일 때 , height 를 검사하는 **중첩 조건문**이다.  
