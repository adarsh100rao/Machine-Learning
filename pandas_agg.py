import numpy as np
import pandas as pd


def pandas_agg():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv')

    print(df.head())

    print(df.body_mass_g.mean())
    print(df.agg({'bill_length_mm': ['min', 'mean'],
                  'flipper_length_mm': ['max', 'mean']
                  }))


def penguins():
    penguin_sample = {'species': {0: 'Adelie', 1: 'Adelie', 2: 'Gentoo', 3: 'Gentoo', 4: 'Adelie',
                                  5: 'Adelie', 6: 'Adelie', 7: 'Gentoo', 8: 'Gentoo', 9: 'Adelie', 10: 'Adelie',
                                  11: 'Adelie', 12: 'Adelie', 13: 'Adelie', 14: 'Gentoo', 15: 'Adelie', 16: 'Adelie',
                                  17: 'Adelie', 18: 'Adelie', 19: 'Adelie'},
                      'sex': {0: 'FEMALE', 1: 'FEMALE', 2: 'FEMALE', 3: 'MALE', 4: 'FEMALE', 5: 'MALE',
                              6: 'MALE', 7: 'MALE', 8: 'MALE', 9: 'MALE', 10: 'FEMALE', 11: 'FEMALE', 12: 'FEMALE',
                              13: 'MALE', 14: 'FEMALE', 15: 'FEMALE', 16: 'FEMALE', 17: 'MALE', 18: 'FEMALE',
                              19: 'MALE'}}

    df = pd.DataFrame(penguin_sample)

    print(df.groupby('sex').agg({'species': 'count'}))


def agg_prob():
    penguin_sample = {'species': {0: 'Gentoo', 1: 'Gentoo', 2: 'Gentoo', 3: 'Adelie', 4: 'Gentoo', 5: 'Gentoo', 6: 'Gentoo', 7: 'Chinstrap', 8: 'Chinstrap', 9: 'Chinstrap', 10: 'Adelie', 11: 'Chinstrap', 12: 'Chinstrap', 13: 'Chinstrap', 14: 'Adelie', 15: 'Adelie', 16: 'Adelie', 17: 'Adelie', 18: 'Adelie', 19: 'Adelie', 20: 'Adelie'}, 'island': {0: 'Biscoe', 1: 'Biscoe', 2: 'Biscoe', 3: 'Biscoe', 4: 'Biscoe', 5: 'Biscoe', 6: 'Biscoe', 7: 'Dream',
                                                                                                                                                                                                                                                                                                                                                            8: 'Dream', 9: 'Dream', 10: 'Dream', 11: 'Dream', 12: 'Dream', 13: 'Dream', 14: 'Torgersen', 15: 'Torgersen', 16: 'Torgersen', 17: 'Torgersen', 18: 'Torgersen', 19: 'Torgersen', 20: 'Torgersen'}, 'body_mass_g': {0: 5650.0, 1: 5600.0, 2: 4400.0, 3: 2900.0, 4: 5350.0, 5: 5550.0, 6: 5100.0, 7: 3800.0, 8: 2700.0, 9: 3675.0, 10: 3950.0, 11: 4000.0, 12: 3600.0, 13: 3775.0, 14: 4200.0, 15: 4100.0, 16: 3150.0, 17: 3300.0, 18: 3700.0, 19: 3550.0, 20: 4250.0}}

    df = pd.DataFrame(penguin_sample)

    # your code here
    print((df.body_mass_g.agg('mean') / 1000).round(3))
    print(df[df.body_mass_g > 4500])


def dates_pandas():
    subscriptions = pd.DataFrame({'day': [12, 15, 8, 21, 23, 9],
                                  'month': [10, 11, 11, 10, 12, 12],
                                  'year': [2022]*6})
    subscriptions['start_date'] = pd.to_datetime(subscriptions)
    print(subscriptions)


def missing_values():
    df = pd.read_csv('data.csv')
    print(df.shape)
    df.dropna(axis=0, inplace=True)
    print(df.shape)


def missing_val_col():
    df = pd.read_csv('data.csv')
    print(df.isnull().sum())


def filling_missing_values():
    df = pd.read_csv('locations.csv')
    mode_location = df['location'].mode()[0]
    mode_building_age = df['building_age'].mode()[0]
    mode_room_num = df['room_num'].mode()[0]

    df['location'].fillna(mode_location, inplace=True)
    df['building_age'].fillna(mode_building_age, inplace=True)
    df['room_num'].fillna(mode_room_num, inplace=True)

    print(df.head(5))


def missing_values_hypercity():
    df = pd.read_csv('hypercity.csv')
    df['height'] = df.groupby('location')['height'].apply(
        lambda x: x.fillna(x.mean()))
    print(df.height.round(decimals=1).sum())


def add_column(df):
    df['sqrt'] = df['1'].apply(lambda x: np.sqrt(x))
    print(df)
