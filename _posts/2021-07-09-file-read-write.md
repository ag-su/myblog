---  
layout: post  
title: python 기초 || 7. 파일 읽기 & 쓰기
image: 
tags:  
categories: python
---

# (7) 파일 읽기 & 쓰기

#### - 읽기모드: r, 쓰기모드(기존 파일 삭제): w, 추가모드(파일 생성 또는 추가): a
- txt 파일: 패스트 캠퍼스 제공

## (1) 파일 읽기

### - 예제 1 


```python
f = open("./resource/review.txt", 'r')
```


```python
content = f.read()
print(content)
print(dir(f))
### 반드시 close 리소스를 반환해야한다. 
f.close()
```

    The film, projected in the form of animation,
    imparts the lesson of how wars can be eluded through reasoning and peaceful dialogues,
    which eventually paves the path for gaining a fresh perspective on an age-old problem.
    The story also happens to centre around two parallel characters, Shundi King and Hundi King,
    who are twins, but they constantly fight over unresolved issues planted in their minds
    by external forces from within their very own units.
    ['_CHUNK_SIZE', '__class__', '__del__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__lt__', '__ne__', '__new__', '__next__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_checkClosed', '_checkReadable', '_checkSeekable', '_checkWritable', '_finalizing', 'buffer', 'close', 'closed', 'detach', 'encoding', 'errors', 'fileno', 'flush', 'isatty', 'line_buffering', 'mode', 'name', 'newlines', 'read', 'readable', 'readline', 'readlines', 'reconfigure', 'seek', 'seekable', 'tell', 'truncate', 'writable', 'write', 'write_through', 'writelines']
    

open의 인자에 'r'을 작성하여 읽기모드로 txt파일을 가져 온다. dir을 출력해서 사용할 수 있는 메소드들을 확인할 수 있다. 

### - 예제 2
- **with문 사용: close 생략 가능**


```python
with open('./resource/review.txt', 'r') as f: 
    c = f.read()
    print(c)
    print(list(c))
    print(iter(c))
```

    The film, projected in the form of animation,
    imparts the lesson of how wars can be eluded through reasoning and peaceful dialogues,
    which eventually paves the path for gaining a fresh perspective on an age-old problem.
    The story also happens to centre around two parallel characters, Shundi King and Hundi King,
    who are twins, but they constantly fight over unresolved issues planted in their minds
    by external forces from within their very own units.
    ['T', 'h', 'e', ' ', 'f', 'i', 'l', 'm', ',', ' ', 'p', 'r', 'o', 'j', 'e', 'c', 't', 'e', 'd', ' ', 'i', 'n', ' ', 't', 'h', 'e', ' ', 'f', 'o', 'r', 'm', ' ', 'o', 'f', ' ', 'a', 'n', 'i', 'm', 'a', 't', 'i', 'o', 'n', ',', '\n', 'i', 'm', 'p', 'a', 'r', 't', 's', ' ', 't', 'h', 'e', ' ', 'l', 'e', 's', 's', 'o', 'n', ' ', 'o', 'f', ' ', 'h', 'o', 'w', ' ', 'w', 'a', 'r', 's', ' ', 'c', 'a', 'n', ' ', 'b', 'e', ' ', 'e', 'l', 'u', 'd', 'e', 'd', ' ', 't', 'h', 'r', 'o', 'u', 'g', 'h', ' ', 'r', 'e', 'a', 's', 'o', 'n', 'i', 'n', 'g', ' ', 'a', 'n', 'd', ' ', 'p', 'e', 'a', 'c', 'e', 'f', 'u', 'l', ' ', 'd', 'i', 'a', 'l', 'o', 'g', 'u', 'e', 's', ',', '\n', 'w', 'h', 'i', 'c', 'h', ' ', 'e', 'v', 'e', 'n', 't', 'u', 'a', 'l', 'l', 'y', ' ', 'p', 'a', 'v', 'e', 's', ' ', 't', 'h', 'e', ' ', 'p', 'a', 't', 'h', ' ', 'f', 'o', 'r', ' ', 'g', 'a', 'i', 'n', 'i', 'n', 'g', ' ', 'a', ' ', 'f', 'r', 'e', 's', 'h', ' ', 'p', 'e', 'r', 's', 'p', 'e', 'c', 't', 'i', 'v', 'e', ' ', 'o', 'n', ' ', 'a', 'n', ' ', 'a', 'g', 'e', '-', 'o', 'l', 'd', ' ', 'p', 'r', 'o', 'b', 'l', 'e', 'm', '.', '\n', 'T', 'h', 'e', ' ', 's', 't', 'o', 'r', 'y', ' ', 'a', 'l', 's', 'o', ' ', 'h', 'a', 'p', 'p', 'e', 'n', 's', ' ', 't', 'o', ' ', 'c', 'e', 'n', 't', 'r', 'e', ' ', 'a', 'r', 'o', 'u', 'n', 'd', ' ', 't', 'w', 'o', ' ', 'p', 'a', 'r', 'a', 'l', 'l', 'e', 'l', ' ', 'c', 'h', 'a', 'r', 'a', 'c', 't', 'e', 'r', 's', ',', ' ', 'S', 'h', 'u', 'n', 'd', 'i', ' ', 'K', 'i', 'n', 'g', ' ', 'a', 'n', 'd', ' ', 'H', 'u', 'n', 'd', 'i', ' ', 'K', 'i', 'n', 'g', ',', '\n', 'w', 'h', 'o', ' ', 'a', 'r', 'e', ' ', 't', 'w', 'i', 'n', 's', ',', ' ', 'b', 'u', 't', ' ', 't', 'h', 'e', 'y', ' ', 'c', 'o', 'n', 's', 't', 'a', 'n', 't', 'l', 'y', ' ', 'f', 'i', 'g', 'h', 't', ' ', 'o', 'v', 'e', 'r', ' ', 'u', 'n', 'r', 'e', 's', 'o', 'l', 'v', 'e', 'd', ' ', 'i', 's', 's', 'u', 'e', 's', ' ', 'p', 'l', 'a', 'n', 't', 'e', 'd', ' ', 'i', 'n', ' ', 't', 'h', 'e', 'i', 'r', ' ', 'm', 'i', 'n', 'd', 's', '\n', 'b', 'y', ' ', 'e', 'x', 't', 'e', 'r', 'n', 'a', 'l', ' ', 'f', 'o', 'r', 'c', 'e', 's', ' ', 'f', 'r', 'o', 'm', ' ', 'w', 'i', 't', 'h', 'i', 'n', ' ', 't', 'h', 'e', 'i', 'r', ' ', 'v', 'e', 'r', 'y', ' ', 'o', 'w', 'n', ' ', 'u', 'n', 'i', 't', 's', '.']
    <str_iterator object at 0x000001DCB12B9320>
    

with를 사용하면 close는 생략 가능하다. with문을 사용하지 않으면 close를 꼭 작성해주어야 한다.

### - 예제 3
- **strip(): 공백 제거**


```python
with open('./resource/review.txt', 'r') as f:
    for c in f:
        print(c.strip())
```

    The film, projected in the form of animation,
    imparts the lesson of how wars can be eluded through reasoning and peaceful dialogues,
    which eventually paves the path for gaining a fresh perspective on an age-old problem.
    The story also happens to centre around two parallel characters, Shundi King and Hundi King,
    who are twins, but they constantly fight over unresolved issues planted in their minds
    by external forces from within their very own units.
    

for문으로 출력할 수 도 있다. strip함수로 공백을 제거한 뒤 출력한다.

### - 예제 4 


```python
with open("./resource/review.txt", 'r') as f:
    content = f.read()
    print(">>>", content)
    content = f.read()  ## 내용 없음
    print('>>>', content)
```

    >>> The film, projected in the form of animation,
    imparts the lesson of how wars can be eluded through reasoning and peaceful dialogues,
    which eventually paves the path for gaining a fresh perspective on an age-old problem.
    The story also happens to centre around two parallel characters, Shundi King and Hundi King,
    who are twins, but they constantly fight over unresolved issues planted in their minds
    by external forces from within their very own units.
    >>> 
    

이렇게 모든 line을 출력하고 나면, 중복해서 line을 불러올 수 없다. 이미 커서가 끝까지 갔기 때문이다. 

### - 예제 5: readline()


```python
with open('./resource/review.txt', 'r') as f:
    line = f.readline()
#     print(line)
    while line:
        print(line, end='####')
        line = f.readline()
```

    The film, projected in the form of animation,
    ####imparts the lesson of how wars can be eluded through reasoning and peaceful dialogues,
    ####which eventually paves the path for gaining a fresh perspective on an age-old problem.
    ####The story also happens to centre around two parallel characters, Shundi King and Hundi King,
    ####who are twins, but they constantly fight over unresolved issues planted in their minds
    ####by external forces from within their very own units.####

readline()은 한줄씩 반환 해 준다. 

### - 예제 6: readlines()


```python
with open("./resource/review.txt", 'r') as f:
        contents = f.readlines()
        print(contents)  # 리스트로 반환 
        print()
        print()
        for c in contents:
            print(c, end='*****')
```

    ['The film, projected in the form of animation,\n', 'imparts the lesson of how wars can be eluded through reasoning and peaceful dialogues,\n', 'which eventually paves the path for gaining a fresh perspective on an age-old problem.\n', 'The story also happens to centre around two parallel characters, Shundi King and Hundi King,\n', 'who are twins, but they constantly fight over unresolved issues planted in their minds\n', 'by external forces from within their very own units.']
    
    
    The film, projected in the form of animation,
    *****imparts the lesson of how wars can be eluded through reasoning and peaceful dialogues,
    *****which eventually paves the path for gaining a fresh perspective on an age-old problem.
    *****The story also happens to centre around two parallel characters, Shundi King and Hundi King,
    *****who are twins, but they constantly fight over unresolved issues planted in their minds
    *****by external forces from within their very own units.*****

readlines는 모든 문장을 리스트에 넣어준다. 

### - 예제 7

형변환을 하고, 리스트에 추가해서 평균을 구하는 예제


```python
score = []

with open('./resource/score.txt', 'r') as f:
    for line in f:
        score.append(int(line))
    print(score)
    
print('---------------------------------')
print('Average: {:6.3}'.format(sum(score) / len(score)))
```

    [95, 78, 92, 89, 100, 66]
    ---------------------------------
    Average:   86.7
    

<br>

## (2) 파일 쓰기

### - 예제 1


```python
with open('./resource/text1.txt', 'w') as f:
    f.write('NiceMan!\n')
```

open의 인자에 'w'를 사용하여 파일 생성 및 글을 작성할 수 있다. 

### - 예제 2: add


```python
with open('./resource/text1.txt', 'a') as f: 
    f.write('Goodboy!\n')
```

open의 인자에 'a'를 사용하여 글씨를 추가할 수 있다. 

 

### - 예제 3
for문을 이용하여 6개의 랜덤 숫자를 각 줄에 한개씩 작성 


```python
from random import randint
```


```python
with open('./resource/text2.txt', 'w') as f:
    for cnt in range(6):
        f.write(str(randint(1, 50)))
        f.write('\n')
```

### - 예제 4
- **writelines: 리스트 -> txt 파일로 저장**


```python
with open('./resource/text3.txt', 'w') as f: 
    list1 = ['kim\n', 'Park\n', 'Cheong\n']
    f.writelines(list1)
```

writelines를 사용하여 리스트를 파일로 저장해줄 수도 있다. 

### - 예제 5
- **프린트 함수로 파일 생성**


```python
with open('./resource/text4.txt', 'w') as f: 
    print('Test1: silver hi!', file=f)
    print('Test2: silver hi!', file=f)
```

print함수의 인자에 file=f 를 넣어주면 파일로 생성이 가능하다. 
