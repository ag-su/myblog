---  
layout: post  
title: python 기초 || 10.2 sqlite 테이블 조회
image: 
tags:  
categories: python
---


# 10. Sqlite 데이터베이스 연동_2
# 테이블 조회


### - Setting


```python
# sqlite
import sqlite3
```


```python
# DB 파일 조회 (없으면 새로 생성)
conn = sqlite3.connect('./resource/database.db') # 본인 DB 경로
```


```python
# 커서 바인딩 
c = conn.cursor()
```

## (2) 테이블 조회

### - 전체 데이터 조회


```python
c.execute("SELECT * FROM users")
```




    <sqlite3.Cursor at 0x1f5c90848f0>



이를 실행하면 아무일도 일어나지 않고, 이제 이 상태에서 **커서 위치**를 변경해가며 데이터를 조회할 수 있다. 

### - 커서 위치 변경

- 1개 로우 선택: **fetchone()**


```python
print('One: \n', c.fetchone())
```

    One: 
     (1, 'KIM', 'silver@naver.com', '010-0000-0000', 'kim.com', '2021-09-21 10:49:23')
    

**fetchone()**을 사용하여 데이터 한개를 선택할 수 있다.

- 지정 로우 선택: **fetchmany( size=n )** -> 리스트 반환


```python
print('Three: \n', c.fetchmany(size=3))
```

    Three: 
     [(2, 'Park', 'Park@gmail.com', '010-1111-1111', 'park.com', '2021-09-21 10:49:23'), (3, 'Lee', 'Lee@naver.com', '010-2222-2222', 'Lee.com', '2021-09-21 10:49:23'), (4, 'Jung', 'jung@gmail.com', '010-3333-3333', 'jung.com', '2021-09-21 10:49:23')]
    

**fetchmany()**의 인자에 **size 개수**를 정해주면 커서의 현재 위치부터 **size개수** 만큼을 가져올 수 있다. 

- 전체 로우 선택: **fetchall()**


```python
print('All: \n', c.fetchall())
```

    All: 
     [(5, 'Yoo', 'yoo@daum.net', '010-4444-4444', 'yoo.net', '2021-09-21 10:49:23')]
    

**fetchall()** 로 현재 커서의 위치 부터 모든 데이터를 가져올 수 있다. 


```python
print('All: \n', c.fetchall())  # 모든 데이터 조회.
```

    All: 
     []
    

5개의 데이터를 모두 조회했으므로 남은 데이터가 없다.

### -  순회

- 순회 1


```python
c.execute("SELECT * FROM users")
```




    <sqlite3.Cursor at 0x1f5c90848f0>




```python
rows = c.fetchall()
for row in rows:
    print('retrieve1: ', row)
```

    retrieve1:  (1, 'KIM', 'silver@naver.com', '010-0000-0000', 'kim.com', '2021-09-21 10:49:23')
    retrieve1:  (2, 'Park', 'Park@gmail.com', '010-1111-1111', 'park.com', '2021-09-21 10:49:23')
    retrieve1:  (3, 'Lee', 'Lee@naver.com', '010-2222-2222', 'Lee.com', '2021-09-21 10:49:23')
    retrieve1:  (4, 'Jung', 'jung@gmail.com', '010-3333-3333', 'jung.com', '2021-09-21 10:49:23')
    retrieve1:  (5, 'Yoo', 'yoo@daum.net', '010-4444-4444', 'yoo.net', '2021-09-21 10:49:23')
    

**fetchall()**을 이용하여 데이터를 반목문으로 순회해 줄 수 있다. 

- 순회 2 


```python
c.execute("SELECT * FROM users")
```




    <sqlite3.Cursor at 0x1f5c90848f0>




```python
for row in c.fetchall():
    print('retrieve2: ', row)
```

    retrieve2:  (1, 'KIM', 'silver@naver.com', '010-0000-0000', 'kim.com', '2021-09-21 10:49:23')
    retrieve2:  (2, 'Park', 'Park@gmail.com', '010-1111-1111', 'park.com', '2021-09-21 10:49:23')
    retrieve2:  (3, 'Lee', 'Lee@naver.com', '010-2222-2222', 'Lee.com', '2021-09-21 10:49:23')
    retrieve2:  (4, 'Jung', 'jung@gmail.com', '010-3333-3333', 'jung.com', '2021-09-21 10:49:23')
    retrieve2:  (5, 'Yoo', 'yoo@daum.net', '010-4444-4444', 'yoo.net', '2021-09-21 10:49:23')
    

순회 1 과 같지만 변수를 만들어 주지 않았을 뿐이다. 

- 순회 3: **가독성이 떨어질 수 있음**


```python
for row in c.execute('SELECT * FROM users ORDER BY id desc'):
    print("retrieve3: ", row)
```

    retrieve3:  (5, 'Yoo', 'yoo@daum.net', '010-4444-4444', 'yoo.net', '2021-09-21 10:49:23')
    retrieve3:  (4, 'Jung', 'jung@gmail.com', '010-3333-3333', 'jung.com', '2021-09-21 10:49:23')
    retrieve3:  (3, 'Lee', 'Lee@naver.com', '010-2222-2222', 'Lee.com', '2021-09-21 10:49:23')
    retrieve3:  (2, 'Park', 'Park@gmail.com', '010-1111-1111', 'park.com', '2021-09-21 10:49:23')
    retrieve3:  (1, 'KIM', 'silver@naver.com', '010-0000-0000', 'kim.com', '2021-09-21 10:49:23')
    

fetchall()을 사용하지 않고 execute로 바로 순회할 수도 있다. 이번엔 ORDER BY 를 사용하여 내림차순 정렬을 하여 순회했다. 

### - WHERE Retrieve

- **방법1:** 튜플


```python
param1 = (3, )
c.execute('SELECT * FROM users WHERE id=?', param1)
print('param1: ', c.fetchone())
print('param1: ', c.fetchall()) # 데이터 없음
```

    param1:  (3, 'Lee', 'Lee@naver.com', '010-2222-2222', 'Lee.com', '2021-09-21 10:49:23')
    param1:  []
    

WHERE을 사용하여 가져올 데이터를 선정할 수 있다. execute의 인자에 튜플로 id값을 넣어주는 방법이다. 

- **방법 2:** format


```python
param2 = 4
c.execute('SELECT * FROM users WHERE id="%s"' %param2)
print('param2: ', c.fetchone())
print('param2: ', c.fetchall())
```

    param2:  (4, 'Jung', 'jung@gmail.com', '010-3333-3333', 'jung.com', '2021-09-21 10:49:23')
    param2:  []
    

% 를 사용하여 id값을 넣어줄 수도 있다. 

- **방법3:** dictionary


```python
c.execute('SELECT * FROM users WHERE id=:ID', {"ID":5})
print('param3: ', c.fetchone())
print("param3: ", c.fetchall())
```

    param3:  (5, 'Yoo', 'yoo@daum.net', '010-4444-4444', 'yoo.net', '2021-09-21 10:49:23')
    param3:  []
    

execute의 인자에 dictionary를 작성하고, key값을 일치지켜 준다. 

- **방법4:** IN & tuple


```python
param4 = (3, 5)
c.execute('SELECT * FROM users WHERE id IN(?, ?)', param4)
print('param4: ', c.fetchall())
```

    param4:  [(3, 'Lee', 'Lee@naver.com', '010-2222-2222', 'Lee.com', '2021-09-21 10:49:23'), (5, 'Yoo', 'yoo@daum.net', '010-4444-4444', 'yoo.net', '2021-09-21 10:49:23')]
    

IN 과 튜플을 사용하여 여러개의 데이터를 가져올 수 있다. 

- **방법5:** IN * format


```python
c.execute('SELECT * FROM users WHERE id IN("%d", "%d")' % (3, 4 ))
print('param5: ', c.fetchall())
```

    param5:  [(3, 'Lee', 'Lee@naver.com', '010-2222-2222', 'Lee.com', '2021-09-21 10:49:23'), (4, 'Jung', 'jung@gmail.com', '010-3333-3333', 'jung.com', '2021-09-21 10:49:23')]
    

IN과 % 를 사용하여 여러개를 가져올 수도 있다. 

- **방법6:** OR & dictionary


```python
c.execute('SELECT * FROM users WHERE id=:id1 OR id=:id2', {'id1': 2, 'id2':5})
print('param6: ', c.fetchall())
```

    param6:  [(2, 'Park', 'Park@gmail.com', '010-1111-1111', 'park.com', '2021-09-21 10:49:23'), (5, 'Yoo', 'yoo@daum.net', '010-4444-4444', 'yoo.net', '2021-09-21 10:49:23')]
    

OR과 인자에 dictionary에 key로 연결시켜 주어 데이터를 가져온다. 

### - Dump 출력 (중요!): conn.iterdump()
- 테이블 생성, 값 삽입 연산들을 한 번에 볼 수 있는 파일 생성


```python
with conn: 
    with open("./resource/dump.sql", 'w') as f: 
        for line in conn.iterdump():
            f.write("%s\n" %line)
        print('Dump Print Complete')
```

    Dump Print Complete
    

이렇게 테이블 생성, 값 삽입 연산들을 한 번에 볼 수 있는 파일이 생성 된다. 

### - 연결 해제
- : with문을 사용했기 때문에 자동 호출된다.


```python
# f.close()
# conn.close()
```
