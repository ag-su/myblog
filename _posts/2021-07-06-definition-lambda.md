---  
layout: post  
title: python 기초 || 4. 함수와 람다
image: 
tags:  
categories: python
---


# 4. 함수와 람다

### 함수 정의방법
- def 함수명(parameter):

   ... code

## (1) 출력 (print)


```python
def hello(world):
    print("Hello", world)
```


```python
hello('Silver !')
hello(777777)
```

    Hello Silver !
    Hello 777777
    

**함수명(parameter)** 의 형태로 함수를 호출할 수 있다. 

## (2) 리턴 (return)


```python
def hello_return(world):
    val = 'Hello' + str(world)
    return val 
```


```python
str = hello_return('Silver!!')
print(str)
```

    HelloSilver!!
    

함수 안에서 **return**하여 **변수**로 **리턴값**을 받아줄 수도 있다. 

## (3) 여러 값 리턴


```python
def func_mul(x):
    y1 = x * 100
    y2 = x * 200
    y3 = x * 300 
    return y1, y2, y3 
```


```python
val1, val2, val3 = func_mul(100)
print(type(val1), val1, val2, val3)
```

    <class 'int'> 10000 20000 30000
    

여러 값을 리턴하여 각각의 변수에 값을 받아줄 수 있다. 

## (4) 리스트 리턴


```python
def func_mul2(x):
    y1 = x * 100
    y2 = x * 200
    y3 = x * 300 
    return [y1, y2, y3]
```


```python
l1 = func_mul2(100)
print(l1, type(l1))
```

    [10000, 20000, 30000] <class 'list'>
    

## (5) 가변 인자 받기 (*args, **kwargs)

### - *args
: 튜플 형태로 받음 (가변 튜플)


```python
def args_func(*args):
    print(args)
```


```python
def args_func2(*args):
    for t in args:
        print(t)
```


```python
def args_func3(*args):
    for i, v in enumerate(args):
        print(i, v)
```


```python
args_func('apple')
print("-------------------------")
args_func('apple', 'banana') 
print("-------------------------")
args_func2('apple', 'banana', 'orange')
print("-------------------------")
args_func3('apple', 'banana', 'orange', 'peach')
```

    ('apple',)
    -------------------------
    ('apple', 'banana')
    -------------------------
    apple
    banana
    orange
    -------------------------
    0 apple
    1 banana
    2 orange
    3 peach
    

### - **kwargs
: dictionary로 받음 (가변 딕셔너리)


```python
def kwargs_func(**kwargs):
    print(kwargs)
```


```python
def kwargs_func2(**kwargs):
    for k, v in kwargs.items():
        print(k, v)
```


```python
kwargs_func(f1 ='apple')
print("----------------------------------------------------------------")
kwargs_func(f1='apple', f2='banana', f3='orange')
print("----------------------------------------------------------------")
kwargs_func2(f1='apple', f2='banana', f3='orange', f4='peach')
```

    {'f1': 'apple'}
    ----------------------------------------------------------------
    {'f1': 'apple', 'f2': 'banana', 'f3': 'orange'}
    ----------------------------------------------------------------
    f1 apple
    f2 banana
    f3 orange
    f4 peach
    

인자에 key = value 형태로 넣어주면 그 개수 대로 딕셔너리 값이 생성되는 것을 볼 수 있다.

### - 전체 혼합


```python
def example_mul(arg1, arg2, *args, **kwargs):
    print(arg1, arg2, args, kwargs)
```

일반, 가변 튜플, 가변 딕셔너리를 모두 받는 함수를 선언한다.  


```python
example_mul(10, 20)
example_mul(10, 20, 30, 'apple', 'banana')
example_mul(10, 20, 30, 'apple', 'banana', 'orange', num1=7, num2=77)
```

    10 20 () {}
    10 20 (30, 'apple', 'banana') {}
    10 20 (30, 'apple', 'banana', 'orange') {'num1': 7, 'num2': 77}
    

## (6) 중첩함수 (클로저) 
- **효율적인 메모리 관리 가능**


```python
def nested_func(num):
    def func_in_func(num):
        print('>>>', num)
    print('in func')
    func_in_func(num + 10000)
```


```python
nested_func(10000)
```

    in func
    >>> 20000
    

nested_fun(10000)으로 함수를 호출하면 print문이 실행되고, func_in_func(10000 + 10000)이 호출이 되어 func_in_func안의 print문이 실행되는 것이다. 

## (7) hint
- **인자 타입과 리턴 타입 명시해주기**


```python
def func_mul3(x: int) -> list: 
    y1 = x * 100 
    y2 = x * 200
    y3 = x * 300 
    return [y1, y2, y3]
```


```python
func_mul3(3)
```




    [300, 600, 900]



인자 x는 int, 리턴타입은 list라고 기입해주어 명확히 사용할 수 있다. 

## (8) 람다 lambda
- **람다식: 메모리 절약, 가독성 향상, 코드 간결**
- **함수: 객체생성 -> 리소스(메모리) 할당 vs 람다: 즉시 실행(Heap초기화) -> 메모리 초기화**

### - 일반 함수: 변수 할당


```python
def mul_10(num: int) -> int:
    return num * 10 
```


```python
val_func = mul_10  # 객체 생성, 메모리 할당 
print(mul_10)
print(val_func)
print(type(val_func))

print(val_func(10))
```

    <function mul_10 at 0x0000023A2BA02EA0>
    <function mul_10 at 0x0000023A2BA02EA0>
    <class 'function'>
    100
    

### - 람다식


```python
lambda_mul_10 = lambda num: num * 10 
print('>>>', lambda_mul_10(10))
```

    >>> 100
    

### - 함수의 인자에 함수 넣기 (람다 사용) 


```python
def func_final(x, y, func):
    print(x * y * func(10))
```

인자에 함수를 넣는 함수를 선언한다. 


```python
func_final(10, 10, lambda_mul_10)
```

    10000
    


```python
func_final(10, 10, lambda x: x * 100000)
```

    100000000
    

이렇게 바로 **lambda** 함수를 사용해주어도 된다. 
