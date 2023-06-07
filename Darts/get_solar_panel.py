import requests
import json
import time
import datetime
import pandas as pd


class SolarPanel():
    def __init__(self, start_time: "2023-04-29", end_time: "2023-04-30", power_or_energy: "energy"):
        self.start_time = start_time
        self.end_time = end_time
        self.API_KEY = 'EDD3ELPSRLR5TD00PHH1G8N4H95EC5OO'
        self.timeUnit = 'QUARTER_OF_AN_HOUR'
        self.siteId = '1934169'
        self.power_or_energy = power_or_energy
        self.df = self.get_data_solar_panel()
    
    def get_df(self):
        return self.df

    def get_data_solar_panel(self):
        # Get data from solar panel
        url = 'https://'
        url += 'monitoringapi.solaredge.com/site/'
        url += self.siteId
        url += '/' + self.power_or_energy + '?' 
        if self.power_or_energy =='energy':
            url += 'timeUnit=' + self.timeUnit
            url += '&endDate='
            url += self.end_time
            url += '&startDate='
            url += self.start_time
        else:
            url += 'startTime='
            url += self.start_time + '%2000:00:00'
            url += '&endTime='
            url += self.end_time + '%2023:59:59'
        url += '&api_key='
        url += self.API_KEY
        print(url)
        response = requests.get(url)
        # print(response.content)
        # Convert response to json
        data = response.json()
        # Convert json to dataframe
        # print(data)

        df = pd.DataFrame(data[self.power_or_energy]['values'])
        # convert Nan to 0
        df = df.fillna(0)
        return df

if __name__ == "__main__":
    start_time = '2023-05-01'
    end_time = '2023-05-30'
    df_solar_long = SolarPanel(start_time, end_time, power_or_energy='power').get_df()
    df_solar_long.to_csv('solar_long.csv')