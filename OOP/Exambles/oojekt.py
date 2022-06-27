class Sensor:
    def __init__(self, pin):
        def getData(self):
            def getPin(self):
                def getTemperature(self):
                    
                    
                    self.pin = pin
                    self.temperature = None








def getData(self):
    self.temperature = simulateData(self.getPin())
    return self.temperature


mySensor = Sensor(1)
while True:
    result = mySensor.getData()
    if result:
        print  % (mySensor.getPin(), result)
    else:
        print  % (mySensor.getPin())
        time.sleep(3)
