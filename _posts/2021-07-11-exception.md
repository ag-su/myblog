---  
layout: post  
title: python 기초 || 8. 예외 종류와 예외 처리
image: 
tags:  
categories: python
---

# 8. 예외 종류와 예외 처리

\* 문법적으로 에러가 없지만, 코드 실행 (런타임) 프로세스에서 발생하는 예외 처리도 중요하다. 

\* (cf) linter : 코드 스타일, 문법 체크 

## (1) 예외 종류

### - SyntaxError: 잘못된 문법


```python
print("Test)
```


      File "<ipython-input-2-9329c5552c3e>", line 1
        print("Test)
                    ^
    SyntaxError: EOL while scanning string literal
    



```python
if True
    pass
```


      File "<ipython-input-3-a2354b54985c>", line 1
        if True
               ^
    SyntaxError: invalid syntax
    



```python
x => y
```


      File "<ipython-input-4-2e326fb65d98>", line 1
        x => y
           ^
    SyntaxError: invalid syntax
    


### - NameError: 참조변수 없음


```python
a = 10 
b = 15
print(c)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-e0eb65479426> in <module>
          1 a = 10
          2 b = 15
    ----> 3 print(c)
    

    NameError: name 'c' is not defined


### - ZerpDivisionError: 0나누기 에러


```python
print(10 / 0)
```


    ---------------------------------------------------------------------------

    ZeroDivisionError                         Traceback (most recent call last)

    <ipython-input-6-65966e0a195f> in <module>
    ----> 1 print(10 / 0)
    

    ZeroDivisionError: division by zero


### - IndexError: 인덱스 범위 오버


```python
x = [10, 20, 30]
print(x[0])
print(x[3])
```

    10
    


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-8-ea3df66951ab> in <module>
          1 x = [10, 20, 30]
          2 print(x[0])
    ----> 3 print(x[3])
    

    IndexError: list index out of range


### - KeyError: 주로 딕셔너리에서 발생


```python
dic = {'name': 'Kim', 'Age': 33, 'city': 'Seoul'}
print(dic['hobby'])
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-10-65fc88a30a7f> in <module>
          1 dic = {'name': 'Kim', 'Age': 33, 'city': 'Seoul'}
    ----> 2 print(dic['hobby'])
    

    KeyError: 'hobby'



```python
print(dic.get('hobby'))  ## 권장: 에러가 나지 않음
```

    None
    

에러가 발생하지 않고 None을 반환하는 get함수 사용을 권장한다. 

### - AttributeError: 모듈, 클래스에 있는 잘못된 속성 사용 시 


```python
import time
print(time.time())
print(time.month())
```

    1631860945.6601243
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-14-828e93408638> in <module>
          1 import time
          2 print(time.time())
    ----> 3 print(time.month())
    

    AttributeError: module 'time' has no attribute 'month'


### - ValueError: 값 에러


```python
x = [1, 5, 9]
x.remove(10)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-15-6ff7fd1f06dd> in <module>
          1 x = [1, 5, 9]
    ----> 2 x.remove(10)
    

    ValueError: list.remove(x): x not in list



```python
x.index(10)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-16-7a80eab74409> in <module>
    ----> 1 x.index(10)
    

    ValueError: 10 is not in list


### - FileNotFoundError


```python
f = open('test.txt', 'r')
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-17-9f3bddd907fb> in <module>
    ----> 1 f = open('test.txt', 'r')
    

    FileNotFoundError: [Errno 2] No such file or directory: 'test.txt'


### - TypeError


```python
x = [1, 2]
y = (1, 2)
z = 'test'

print(x + y)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-18-87289a4638c3> in <module>
          3 z = 'test'
          4 
    ----> 5 print(x + y)
    

    TypeError: can only concatenate list (not "tuple") to list



```python
print(x + z)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-19-5bf2e5d9911c> in <module>
    ----> 1 print(x + z)
    

    TypeError: can only concatenate list (not "str") to list



```python
print(x + list(y))  # 형변환을 해준다
```

    [1, 2, 1, 2]
    

리스트는 리스트끼리만 합칠 수 있다는 에러가 났고, 위와 같이 list로 형변환을 해주면 연산이 가능해진다. 

<br>

\* 항상 예외가 발생하지 않을 것으로 가정하고 먼저 코딩한다. 

\-\> 그 후 런타임 예외 발생 시 예외 처리 코딩 권장 ( EAFP 코딩 스타일 ) 

## (2) 예외 처리

- **try: 에러가 발생할 가능성이 있는 코드 실행(여기서 에러가 나면 except로 간다)**
- **except: 에러명**
- **else: 에러가 발생하지 않았을 경우 실행**
- **finally: 항상 실행**

### - 예제 1


```python
name = ['kim', 'Lee', 'Park']

try:
    z = 'kim'
    x = name.index(z)
    print('{} found! it in name'.format(z, x+1))

except ValueError:
    print('Not found it! - Occured ValueError!')

else:
    print('OK! else!')

```

    kim found! it in name
    OK! else!
    


```python
name = ['Kim', 'Lee', 'Park']

try:
    z = 'Cho'
    x = name.index(z)
    print('{} found! it in name'.format(z, x+1))

except ValueError:
    print('Not found it! - occured ValueError!')
    
else:
    print('ok! else')
```

    Not found it! - occured ValueError!
    


오류가 발생하였기 때문에 except 블럭이 실행되었다. 

### - 예제 2 


```python
name = ['Kim', 'Lee', 'Park']

try:
    z = 'jim'
    x = name.index(z)
    print("{} found! it in name".format(z, x+1))
    
except:
    print('Not found it! - Occured Error!')

else: 
    print("OK, else!")
```

    Not found it! - Occured Error!
    

### - 예제 3 


```python
try:
    z = 'Kim'
    x = name.index(z)
    print("{} found! it in name".format(z, x+1))
except:
    print("Not found it! - Occured Error!")
else:
    print("Ok, else!")
finally:
    print("finally ok!")
```

    Kim found! it in name
    Ok, else!
    finally ok!
    

### - 예제 4
- **예외처리는 하지 않지만, 무조건 수행되는 코딩 패턴**


```python
try: 
    print("try")
finally:
    print('OK finally!!')
```

    try
    OK finally!!
    

### - 예제 5
- **여러 개의 except 블럭**


```python
try:
#     z = 'Kim'
    z = 'Cho'
    x = name.index(z)
    print('{} found! it in name'.format(z, x+1))
# except Exception as f:  # 이걸 처음에 두면 안된다 ! ( 모든 에러를 포함하기 때문 ) 
#     print(f)
except IndexError:
    print("Not found it! - Occured IndexError!")
except ValueError:
    print("Not Found it! - Occured ValueError!")
except Exception:
    print("Not found it! - Occured Error!")
else:
    print('OK! else!')
finally:
    print('finally ok!')
```

    Not Found it! - Occured ValueError!
    finally ok!
    

위와 같이 에러를 여러 개 처리할 수도 있다. 이 때 **Exception**을 가장 첫 번째 넣어주면 밑에 자세한 에러는 나지 않기 때문에 **위치 선정을 주의**하여야 한다. ( **Exception이 큰 범위**이기 때문 ) 

### - 예제 6
- **예외 발생: raise (raise 키워드로 예외 직접 발생)**


```python
try: 
#     a = 'Kim'
    a = 'Jim'
    if a == 'Kim':
        print('Ok!')
    else:
        raise ValueError
except ValueError:
    print("문제 발생!")
except Exception as f:
    print(f)
else:
    print('ok!')
```

    문제 발생!
    

**raise**를 이용하여 직접 예외를 발생시킬 수도 있다. 이 예제에서는 **ValueError**를 발생시키라고 하였으므로 **except ValueError**부분이 실행되었다.
