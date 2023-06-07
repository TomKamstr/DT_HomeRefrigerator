import datetime
import serial
import time
import pandas as pd
import os
import csv

class WaterTempSensors():
    def __init__(self):
        self.folder_name = self.make_folder_of_today()
        self.read_sensors()

    def make_folder_of_today(self):
        now = datetime.datetime.now()
        year = now.year
        month = now.month
        day = now.day
        hour = now.hour
        minute = now.minute
        second = now.second
        folder_name = str(year) + "_" + str(month) + "_" + str(day) + "_" + str(hour) + "_" + str(minute) + "_" + str(second)
        folder_name = os.getcwd() + "/Logged_Data_WaterTemps/" + folder_name
        # use try and except to avoid error when the folder already exists
        try:
            os.mkdir(folder_name)
            return folder_name  

        except OSError:
            print ("Creation of the directory %s failed" % folder_name)
            return False
    
    def read_sensors(self):
        file_name = self.folder_name + "/data_water_temps.csv"
        with open(file_name, 'w') as file:
            fieldnames = ['time', 'temp1', 'temp2', 'temp3'] 
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
        # TODO: Aanpassen op Pi koelkast freezerdata
        ser = serial.Serial('/dev/ttyACM2', 9600)
        ser.flush()
        list_temp1 = []
        list_temp2 = []
        list_temp3 = []
        while True:
            if ser.in_waiting > 0:
                try: 
                    line = ser.readline().decode('utf-8').rstrip()
                    line_splitted = line.split(", ")
                    # print(line_splitted)
                    list_temp1.append(float(line_splitted[0])) 
                    list_temp2.append(float(line_splitted[1]))
                    list_temp3.append(float(line_splitted[2]))
                    if len(list_temp1) > 15:
                        with open(file_name, 'a') as csvfile:
                            fieldnames = ["time", "temp1", "temp2", "temp3"]
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            temp1 = sum(list_temp1) / len(list_temp1)
                            temp2 = sum(list_temp2) / len(list_temp2)
                            temp3 = sum(list_temp3) / len(list_temp3)
                            writer.writerow({'time': datetime.datetime.now(), 'temp1': temp1, 'temp2': temp2, 'temp3': temp3})
                            print("Exported: ", temp1, temp2, temp3)
                            list_temp1 = []
                            list_temp2 = []
                            list_temp3 = []
                except serial.serialutil.SerialException:
                    print("Serial exception")
                    continue
                except IndexError:
                    print("Index error")
                    continue
        ser.close()

if __name__ == "__main__":
    WaterTempSensors()