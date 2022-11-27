---  
layout: post  
title: python 기초 || 6. 모듈과 패키지 module & package
image: 
tags:  
categories: python
---

# 6. 모듈과 패키지 module & package

- **폴더: 패키지, 파일: 모듈 이라고 할 수 있음** 

- 패키지: pkg 폴더
- 모듈: __ init __.py, fibonacci.py, calculations.py, prints.py 파일 생성

#### 파일 소개

- **pkg/__ init __.py**  

\* 용도- 해당 디렉토리가 **패키지**임을 선언한다.  
\* **Python 3.x** 버전은 파일이 없어도 패키지를 인식 하지만 , 하위 호환을 위해 생성해 놓는 것을 추천한다. 

- **pkg/fibonacci.py**: 피보나치 수열 출력 및 리스트 출력 함수가 선언되어 있음


```python
class Fibonacci:
    def __init__(self, title='fibonacci'):
        self.title = title 

    def fib(n):  # 피보나치 출력
        a, b = 0, 1 
        while a < n:
            print(a, end=' ')
            a, b = b, a + b 
        print()

    def fib2(n): # 피보나치 리스트 리턴 
        result = []
        a, b = 0, 1 
        while a < n:
            result.append(a)
            a, b = b, a + b 

        return result
```

- **pkg/calculations.py**: 단순 계산 함수 (더하기, 곱하기, 나누기) 


```python
def add(l, r):
    return l + r

def mul(l, r):
    return l * r

def div(l, r):
    return l / r
```

- **pkg/prints.py**: 단순 출력문 함수 

단위실행으로 다른 파일에서 이 모듈을 실행했을 때 출력되지 않고, 현재 파일에서만 실행이 되도록 하는 if문을 선언한다. 

 


```python
def prt1():
    print("I'm NiceGirl!")


def prt2():
    print("I'm GoodGirl!")


# 단위 실행 (독립적으로 파일 실행) : 함수가 잘 만들어 졌는지 확인하기 위해 이 파일에서만 실행하도록 설정 
if __name__ == "main":
    prt1()
    prt2()
```

## (1) 예제 1: 클래스


```python
from pkg.fibonacci import Fibonacci
```

위와 같이 클래스를 import해 올 수 있다. 


```python
Fibonacci.fib(300)
```

    0 1 1 2 3 5 8 13 21 34 55 89 144 233 
    

인스턴스 메소드가 아닌 클래스 메소드였으므로 클래스 자체로 접근이 가능하다. 


```python
print(Fibonacci.fib2(400))
print(Fibonacci().title)  # 인스턴스 변수: 인스턴스화 시켜야 함 
```

    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    fibonacci
    

title은 인스턴스 변수기 때문에 인스턴스화 시켜 가져올 수 있다. 

## (2) 예제 2: 클래스 2 


```python
from pkg.fibonacci import * 
## 모듈의 모든 클래스를 가져오는 방법: 메모리를 많이 차지하므로 권장 x, 
## 필요한 것만 가져오는 것이 효율적임
```


```python
Fibonacci.fib(500)
print(Fibonacci.fib2(600))
print(Fibonacci().title)
```

    0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    fibonacci
    

모듈의 모든 클래스를 가져오는 방법이다. 이 방법은 메모리를 많이 차지하므로 권장되지 않는다. 필요한 것만 불러오는 것이 효율적이다. 

## (3) 예제 3: 클래스 3 -> 권장하는 방법




```python
from pkg.fibonacci import Fibonacci as fb
```

많이 사용되는 방법이고, 이와 같이 사용하는 것이 권장된다. 


```python
fb.fib(1000)
print(fb.fib2(1600))
print(fb().title)
```

    0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
    fibonacci
    

## (4) 예제4: 함수 


```python
import pkg.calculations as c
```


```python
print(c.add(10, 100))
print(c.mul(10, 100))
```

    110
    1000
    

## (5) 예제5: 함수 - 권장하는 방법
- **필요한 만큼만 사용하는 것이 좋은 습관이다.**


```python
from pkg.calculations import div as d 
```


```python
print(int(d(100, 10)))
```

    10
    

이렇게 필요한 함수만 불러와서 사용할 수 있다. 

## (6) 예제 6


```python
import pkg.prints as p
import builtins # 파이썬에서 기본적으로 제공하는 함수들 
```


```python
p.prt1()
p.prt2()
print()
print(dir(builtins)) # pkg.prints 에서 사용한 __name__ 가 builtins에 들어가 있는 것을 볼 수 있다. 
```

    I'm NiceGirl!
    I'm GoodGirl!
    
    ['ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException', 'BlockingIOError', 'BrokenPipeError', 'BufferError', 'BytesWarning', 'ChildProcessError', 'ConnectionAbortedError', 'ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError', 'DeprecationWarning', 'EOFError', 'Ellipsis', 'EnvironmentError', 'Exception', 'False', 'FileExistsError', 'FileNotFoundError', 'FloatingPointError', 'FutureWarning', 'GeneratorExit', 'IOError', 'ImportError', 'ImportWarning', 'IndentationError', 'IndexError', 'InterruptedError', 'IsADirectoryError', 'KeyError', 'KeyboardInterrupt', 'LookupError', 'MemoryError', 'ModuleNotFoundError', 'NameError', 'None', 'NotADirectoryError', 'NotImplemented', 'NotImplementedError', 'OSError', 'OverflowError', 'PendingDeprecationWarning', 'PermissionError', 'ProcessLookupError', 'RecursionError', 'ReferenceError', 'ResourceWarning', 'RuntimeError', 'RuntimeWarning', 'StopAsyncIteration', 'StopIteration', 'SyntaxError', 'SyntaxWarning', 'SystemError', 'SystemExit', 'TabError', 'TimeoutError', 'True', 'TypeError', 'UnboundLocalError', 'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning', 'UserWarning', 'ValueError', 'Warning', 'WindowsError', 'ZeroDivisionError', '__IPYTHON__', '__build_class__', '__debug__', '__doc__', '__import__', '__loader__', '__name__', '__package__', '__spec__', 'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex', 'copyright', 'credits', 'delattr', 'dict', 'dir', 'display', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format', 'frozenset', 'get_ipython', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'license', 'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip']
    

builtins에는 파이썬에서 기본적으로 제공하는 함수들을 볼 수 있는데, prints.py에서 사용했던 \__name\__이 들어가 있는 것을 볼 수 있다. 
