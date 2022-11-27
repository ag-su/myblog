---  
layout: post  
title: python 기초 || 10.1 sqlite 테이블 생성, 삽입, 삭제
image: 
tags:  
categories: python
---



# 10. Sqlite 데이터베이스 연동_1
# 테이블 생성, 데이터 삽입과 삭제


### - setting 
- **library import**


```python
import sqlite3
import datetime
```

- **SQliteDatabaseBrowserPortable.exe** 설치하기 

: 직관적인 데이터베이스 관리 가능

![image.png]({{site.baseurl}}/images/sqlite/sqlite1.png)

- **삽입날짜 변수 생성**


```python
now = datetime.datetime.now()
print("now: ", now)

nowDateTime = now.strftime('%Y-%m-%d %H:%M:%S')
print('nowDateTime: ', nowDateTime)
```

    now:  2021-09-20 17:36:58.730291
    nowDateTime:  2021-09-20 17:36:58
    

데이터베이스에 넣어줄 변수를 사전 생성한다.  

- **sqlite 살펴보기**


```python
print('sqlite3.version: ', sqlite3.version)
print('sqlite3.sqlite_version', sqlite3.sqlite_version)
```

    sqlite3.version:  2.6.0
    sqlite3.sqlite_version 3.33.0
    

## (1) 테이블 생성

### - DB 생성 & Auto Commit(Rollback) 설정 
- isolation_level=None: AutoCommit


```python
conn = sqlite3.connect('./resource/database.db', isolation_level=None)
```

![image.png]({{site.baseurl}}/images/sqlite/sqlite2.png)

**DB**를 생성하고 연결해 준다. **DB**에 명령할 때 **커밋(commit)**을 꼭 해주어야 하는데 , **isolation_level=None**으로 설정하면 **AutoCommit**이 된다. 

 

그리고 아까 다운받았던 **SQliteDatabaseBrowserPortable.exe** 에서 **[ 데이터베이스 열기 - database.db ]** 를 실행한다.

### - Cursor 


```python
c = conn.cursor()
print('Cursor Type: ', type(c))
```

    Cursor Type:  <class 'sqlite3.Cursor'>
    

**cursor** 연결을 해준다. 이 커서로 데이터베이스에 명령을 할 수 있다. 

### - 테이블 생성 CREATE TABLE
- **data type: TEXT, NUMERIC, INTEGER, REAL BLOB ... )**


```python
c.execute('CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT, email TEXT, phone TEXT, website TEXT, regdate TEXT)')
```




    <sqlite3.Cursor at 0x193695339d0>



테이블을 생성하기 위해서 CREATE TABLE 명령어를 사용한다. 그다음 **테이블이름(속성 타입)** 이러한 형태로 넣어준다. **id 속성**을 **PRIMARY KEY**로 설정해 주었다.

![image.png]({{site.baseurl}}/images/sqlite/sqlite3.png)

이렇게 **users** 테이블이 생성되었다.

### - 데이터 삽입 INSERT INTO

- 방법 1


```python
c.execute("INSERT INTO users VALUES(1, 'KIM', 'silver@naver.com', '010-0000-0000', 'kim.com', ?)", (nowDateTime,))
```




    <sqlite3.Cursor at 0x193695339d0>



- 방법 2


```python
c.execute("INSERT INTO users(id, username, email, phone, website, regdate) VALUES(?, ?, ?, ?, ?, ?)", (2, 'Park', 'Park@gmail.com', '010-1111-1111', 'park.com', nowDateTime))
```




    <sqlite3.Cursor at 0x193695339d0>



- INSERT INTO 테이블이름 VALUES() 

- INSERT INTO 테이블이름() VALUES() 

방법 1은 바로 VALUES안에 값을 넣고, 변수를 넣을 경우 ?로 작성한 후 execute의 인자에서 튜플안에 변수를 넣어준다.  방법 2는 테이블이름() 안에 속성 이름을 명시적으로 나열하고 , VALUES() 안에 모두 ? 로 작성해 주고, execute의 인자에서 튜플 안에 값 또는 변수를 넣어준다. 

![image.png]({{site.baseurl}}/images/sqlite/sqlite4.png)

새로고침 하면 위 화면과 같이 값이 추가된 것을 볼 수 있다.  

### - Many 삽입 (튜플, 리스트): executemany


```python
userList = (
    (3, 'Lee', 'Lee@naver.com', '010-2222-2222', 'Lee.com', nowDateTime),
    (4, 'Jung', 'jung@gmail.com', '010-3333-3333', 'jung.com', nowDateTime),
    (5, 'Yoo', 'yoo@daum.net', '010-4444-4444', 'yoo.net', nowDateTime)
)

c.executemany("INSERT INTO users(id, username, email, phone, website, regdate) VALUES(?, ?, ?, ?, ?, ?)", userList)
```




    <sqlite3.Cursor at 0x193695339d0>



![image.png]({{site.baseurl}}/images/sqlite/sqlite5.png)

**excutemany**의 두번째 인자에 튜플이나 리스트를 넣어주면 한꺼번에 데이터 삽입이 가능하다. 

### - 테이블 데이터 삭제: DELETE FROM 


```python
c.execute('DELETE FROM users')
```




    <sqlite3.Cursor at 0x193695339d0>



이렇게 작성하면 users 테이블의 모든 데이터가 삭제된다. 

- **지워진 개수 반환하고 삭제하기**


```python
userList = (
    (3, 'Lee', 'Lee@naver.com', '010-2222-2222', 'Lee.com', nowDateTime),
    (4, 'Jung', 'jung@gmail.com', '010-3333-3333', 'jung.com', nowDateTime),
    (5, 'Yoo', 'yoo@daum.net', '010-4444-4444', 'yoo.net', nowDateTime)
)

c.executemany("INSERT INTO users(id, username, email, phone, website, regdate) VALUES(?, ?, ?, ?, ?, ?)", userList)

print("users db delete: ", c.execute('DELETE FROM users').rowcount)
```

    users db delete:  3
    

**rowcount**를 사용하여 개수를 반환해줄 수 있다. 

### - 커밋 & 롤백
- isolation_level=None: Autocommit


```python
# conn.commit()
```


```python
# conn.rollback()
```

본 포스팅에서는 오토커밋으로 설정 해놨으므로 주석처리 한다. 

### - 접속 해제
- 마무리할 때 꼭 접속해제를 해야한다. 


```python
conn.close()
```
