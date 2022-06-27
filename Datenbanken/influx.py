from influxdb import InfluxDBClient
from influxdb import DataFrameClient


# using Http
client = InfluxDBClient(database='mydb')
client = InfluxDBClient(host='127.0.0.1', port=8086, database='mydb')
client = InfluxDBClient(host='127.0.0.1', port=8086, username='pi', password='kp02101927', database='mydb')
# using UDP
client = InfluxDBClient(host='127.0.0.1', database='mydb', use_udp=True, udp_port=4444)

