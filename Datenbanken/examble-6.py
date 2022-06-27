# import stuff
from influxDB import influxdb-client
from datetime import datetime
from datetime import * 

# importlib.reload(), z.B. import importlib; importlib.reload(modulename). 

user = 'pi'
password = 'kp02101927'
dbname = 'mydb'
dbuser = 'admin'
dbuser_password = 'kp02101927'
# Setup database
client = InfluxDBClient('localhost', '8086', 'spee',
                        'kp02101927', 'mydb', dbuser, dbuser_password)
client.create_database('mydb')
client.get_list_database()
client.switch_database('mydb')
client.get_list_users()

# Setup payload


json_payload = []
data = {
    "measurement": "sensor",
    "tags": {
        "ticker": "TLSb"
    },
    "time": datetime.today(),
    "fields": {
        'tag': 638.34,
        'nacht': 567.78
    }
}

json_payload.append(data)

# send our payload
client.write_points(json_payload)

client = InfluxDBClient(host='127.0.0.1', port=8086,
                        username='pi', password='kp02101927', database='mydb')

points = client.query('SELECT * from  cpu').get_points()
for point in points:
    print(points)
points = client.query('SELECT * from  stocks').get_points()
for point in points:
    print(points)
points = client.query('SELECT * from  sensor').get_points()
for point in points:
    print(points)
