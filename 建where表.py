
def buildwheretable():
    db = mysql.connect(host="localhost", user="root", password="123456789", db="test", port=3306, charset='utf8')
    cur = db.cursor()
    sql="insert into `where`(location) values(%s)"
    try:
        for s in where:
            print(s)
            cur.execute(sql % ('\''+s+'\''))
        db.commit()
    except Exception as e:
        print(e)
        db.rollback()
    cur.close()
    db.close()