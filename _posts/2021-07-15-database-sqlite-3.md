---  
layout: post  
title: python 기초 || 10.3 sqlite 테이블 수정 및 삭제
image: 
tags:  
categories: python
---
 

# 10. Sqlite 데이터베이스 연동_3
# 테이블 데이터 수정 및 삭제


### - settings


```python
import sqlite3  # sqlite 모듈 import 
```


```python
conn = sqlite3.connect('./resource/database.db')  # DB 연결 (생성)/(파일)
```


```python
c = conn.cursor()  # Cursor 연결 
```

이번에는 **Autocommit** 설정을 하지 않았기 때문에 코드 뒤에 **conn.commit()** 를 항상 선언해야 한다. (글에서는 생략) 

## (3) 테이블 데이터 수정 및 삭제

### - 데이터 수정: UPDATE

- **1) 방법1:** 튜플


```python
c.execute("UPDATE users SET username = ? WHERE id = ?", ('niceman', 2))
```




    <sqlite3.Cursor at 0x24b8bf44c00>



![image.png]({{site.baseurl}}/images/sqlite/sqlite6.png)

id = 2 인 username을 niceman으로 변경하는 코드이다. 

- **2) 방법2:** 딕셔너리


```python
c.execute("UPDATE users SET username=:name WHERE id=:ID", {'name': 'goodgirl', 'ID': 5})
```




    <sqlite3.Cursor at 0x24b8bf44c00>



![image.png]({{site.baseurl}}/images/sqlite/sqlite7.png)

딕셔너리로 key를 연결하여 수정할 수도 있다. 

- **2) 방법3:** format


```python
c.execute("UPDATE users SET username='%s' WHERE id='%s'" % ('badboy', 3))
```




    <sqlite3.Cursor at 0x24b8bf44c00>



![image.png]({{site.baseurl}}/images/sqlite/sqlite8.png)

% 를 이용하여 연결해 줄 수도 있다. 

- 중간 데이터 확인 1


```python
for user in c.execute("SELECT * FROM users"):
    print(user)
```

    (1, 'KIM', 'silver@naver.com', '010-0000-0000', 'kim.com', '2021-09-21 10:49:23')
    (2, 'niceman', 'Park@gmail.com', '010-1111-1111', 'park.com', '2021-09-21 10:49:23')
    (3, 'badboy', 'Lee@naver.com', '010-2222-2222', 'Lee.com', '2021-09-21 10:49:23')
    (4, 'Jung', 'jung@gmail.com', '010-3333-3333', 'jung.com', '2021-09-21 10:49:23')
    (5, 'goodgirl', 'yoo@daum.net', '010-4444-4444', 'yoo.net', '2021-09-21 10:49:23')
    

이렇게 데이터 순회를 통해 바뀐 데이터들을 파이썬에서 직접 확인 할 수 있다. 

### - 데이터 삭제

- **1)** Row Delete


```python
# 방법 1 
c.execute("DELETE FROM users WHERE id=?", (2, ))
```




    <sqlite3.Cursor at 0x24b8bf44c00>




```python
# 방법 2 
c.execute("DELETE FROM users WHERE id=:ID", {'ID': 5})
```




    <sqlite3.Cursor at 0x24b8bf44c00>




```python
# 방법 3
c.execute("DELETE FROM users WHERE id='%s'" % 4)
```




    <sqlite3.Cursor at 0x24b8bf44c00>



![image.png]({{site.baseurl}}/images/sqlite/sqlite9.png)

삭제도 같은 3가지의 방법으로 원하는 행을 선택하여 삭제할 수 있다. 

- 중간 데이터 확인 1


```python
for user in c.execute("SELECT * FROM users"):
    print(user)
```

    (1, 'KIM', 'silver@naver.com', '010-0000-0000', 'kim.com', '2021-09-21 10:49:23')
    (3, 'badboy', 'Lee@naver.com', '010-2222-2222', 'Lee.com', '2021-09-21 10:49:23')
    

- **2)** 테이블 전체 데이터 삭제


```python
print("users db deleted: ", conn.execute("DELETE FROM users").rowcount, 'rows')
```

    users db deleted:  2 rows
    

![image.png]({{site.baseurl}}/images/sqlite/sqlite10.png)

모두 삭제되었다.

- **커밋:** 변경사항을 적용할 때마다 선언해야 한다. 


```python
conn.commit()
```

- **접속 해제:** with문을 사용하지 않으면 접속 해제를 꼭 해주어야 한다. 


```python
conn.close()
```
