---  
layout: post  
title: python 기초 || 2.1 datatype - 숫자형과 문자열
image: 
tags:  
categories: python
---


# 2. Data Type : 숫자형, 문자열

### 데이터 타입 한눈에 보기 


```python
v_str = "Niceman"
v_bool = True
v_float = 7.77
v_int = 7
v_dict = {
    'name' : 'Silver',
    'age' : 25
}
v_list = [7, 8, 9]
v_tuple = 7, 8, 9
v_set = {7, 8, 9}
```


```python
print(type(v_str))
print(type(v_bool))
print(type(v_float))
print(type(v_int))
print(type(v_dict))
print(type(v_list))
print(type(v_tuple))
print(type(v_set))
```

    <class 'str'>
    <class 'bool'>
    <class 'float'>
    <class 'int'>
    <class 'dict'>
    <class 'list'>
    <class 'tuple'>
    <class 'set'>
    

## (1) 숫자형

[숫자형 연산자 참고](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) 에서 숫자형 연산자에 관한 내용을 확인할 수 있다. 


```python
# 변수 선언 
i1 = 7 
i2 = 77
f1 = 4.321
f2 = .999

print(type(i1))
print(type(i2))
print(type(f1))
print(type(f2))
```

    <class 'int'>
    <class 'int'>
    <class 'float'>
    <class 'float'>
    


```python
print(i1 * i2)
print(f1 ** f2)
```

    539
    4.314680898380055
    


```python
result = i1 + f2
print(result, type(result))  # int + float => float
```

    7.999 <class 'float'>
    

int + float를 하면 자동적으로 float 타입이 된다. 

- 숫자형 형변환 (유연한 형변환 가능 )


```python
print(int(result))
print(float(i1))
print(complex(7))  # 복소수 
print(int(True))
print(int(False))
print(int('3'))
print(complex(False))
```

    7
    7.0
    (7+0j)
    1
    0
    3
    0j
    

True, False는 각각 정수형으로 변환하면 1, 0 이 된다. 

- 수치 연산 함수 

[수치 연산 함수](https://docs.python.org/3/library/math.html) 에서 수치 연산에 관한 함수를 확인할 수 있다. 


```python
print(abs(-777))  # 절댓값 
n, m = divmod(100, 7)  # 몫, 나머지 
print(n, m)
```

    777
    14 2
    

**abs()** 함수는 절댓값을 씌워주는 함수이고, **divmod()**함수는 몫과 나머지를 동시에 반환해 준다. 


```python
import math 
print(math.ceil(7.1))  # 괄호 안의 숫자보다 크면서 가장 작은 정수 
print(math.floor(7.1))  # 괄호 안의 숫자보다 작으면서 가장 큰 정수 
```

    8
    7
    

- math.ceil() : 인자에 들어간 숫자보다 크면서 가장 작은 정수 
- math.floor() : 인자에 들어간 숫자보다 작으면서 가장 큰 정수 

## (2) 문자열


```python
str1 = "I am a geak!"
str2 = "NiceGirl"
str3 = ""
str4 = str()
```


```python
print(len(str1), len(str2), len(str3), len(str4), sep='\n')  # 공백 포함 글자 수 
```

    12
    8
    0
    0
    

공백을 포함하여 문자열의 개수를 세는 len() 함수

- 이스케이프 문자 사용 


```python
escape1 = "Do you have a \"big apple\"?"
print(escape1)
escape2 = "Tab1\tTab2\tTab3\tTab4"
print(escape2)
```

    Do you have a "big apple"?
    Tab1	Tab2	Tab3	Tab4
    

같은 종류 따옴표를 안에 넣기 위하여 이스케이프 문자를 사용한다. 

- Raw String


```python
raw_str1 = r"C:\Python\Test\..."
print(raw_str1)
raw_str2 = r"\\a\\b"
print(raw_str2)
```

    C:\Python\Test\...
    \\a\\b
    

rwa string으로 이스케이프문 없이 그대로 출력이 가능하다. 

- 멀티라인 : 다음줄과 이어짐


```python
multi = \
"""
문자열
멀티라인
테스트
입니다
"""

print(multi)
```

    
    문자열
    멀티라인
    테스트
    입니다
    
    

- 문자열 연산


```python
str_o1 = "*"
str_o2 = "abc"
str_o3 = "def"
str_o4= "NiceGirl"
```


```python
print(str_o1 * 100)
print(str_o2 + str_o3)
print('g' in str_o4)
print('G' in str_o4)
print('z' not in str_o4)
```

    ****************************************************************************************************
    abcdef
    False
    True
    True
    

문자열에 숫자를 곱하거나, 문자열끼리 더하는 연산이 가능하다. 

- 문자열 형변환


```python
str1 = str(77) + 'A'
str2 = str(10.4)
print(str1, type(str1))
print(str2, type(str2))
```

    77A <class 'str'>
    10.4 <class 'str'>
    

- 문자열 함수

[문자열 함수](https://www.w3schools.com/python/python_ref_string.asp)에서 모든 문자열 함수를 확인할 수 있고, 글에서는 자주 쓰이는 함수만 사용해 본다. 


```python
a = "niceman"
b = "orange"
```


```python
print(a.islower())  # 소문자인지 : Bool 반환
print(b.endswith('e'))  # e로 끝나는지 : Boot 반환 
print(a.capitalize())  # 첫 글자를 대문자로 
print(a.replace('nice', "good"))  # nice 를 good으로 대체
print(list(reversed(b)))  # 뒤집어서 리스트로 반환 
```

    True
    True
    Niceman
    goodman
    ['e', 'g', 'n', 'a', 'r', 'o']
    

- 문자열 슬라이싱


```python
a = "nicegirl" 
b = "orange" 
```


```python
print(a[0:3])
print(a[0:4])
print(a[0:len(a)])
print(a[:4])
print(b[0:4:2])
print(b[1:-2])
print(b[::-1])
```

    nic
    nice
    nicegirl
    nice
    oa
    ran
    egnaro
    

 a[0:n] 일 때 0번째 글자부터 n-1번째 글자까지를 반환한다. 
