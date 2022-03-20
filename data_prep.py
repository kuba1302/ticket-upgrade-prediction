import pandas as pd 
import numpy as np
import datetime
from pathlib import Path
import os


import matplotlib.pyplot as plt
import seaborn as sns 




def read_all_csv(data_path, data_type='train', model_type='predict_upgrade'):
    bkg_drop_cols = ['UPGRADED_FLAG', 'UPGRADE_TYPE', 'UPGRADE_SALES_DATE']
    bkg = pd.read_csv(data_path / f'BKG_{data_type}.csv', sep=';')
    if data_type == 'train': 
        bkg = bkg.drop(columns=bkg_drop_cols)
    tkt = pd.read_csv(data_path / f'TKT_{data_type}.csv', sep=';')
    fcp = pd.read_csv(data_path / f'FCP_{data_type}.csv', sep=';')
    if model_type == 'predict_when_upgrade': 
        fcp = fcp[fcp['UPGRADED_FLAG'] == 'Y']
    df = fcp.merge(tkt, on='TICKET_NUMBER', how='left')
    df = df.merge(bkg, on='BOOKING_ID', how='left')
    return df 

def clean_df(data, data_path, data_type='train', model_type='predict_upgrade'):
    drop_cols = [
        "TICKET_NUMBER",
        "ORIGIN_AIRPORT_CODE",
        "DESTINATION_AIRPORT_CODE",
        "SALES_DATE",
        "FLIGHT_DATE_LOCAL",
        "MARKETING_CARRIER",
        "OPERATIONAL_CARRIER",
        "BOOKED_CLASS",
        "AIRCRAFT_TYPE",
        "FARE_BASIS",
        "BOOKING_ID",
        "ORIGINAL_TICKET_NUMBER",
        "SEGMENTS",
        "FLIGHT_COUPONS",
        "FORM_OF_PAYMENT",
        "CURRENCY",
        "TOTAL_PRICE",
        "LOYAL_CUSTOMER_ID",
        "LOYAL_CUSTOMER_DATE_OF_BIRTH",
        "LOYAL_CUSTOMER_REGISTERED_DATE",
        "SALES_DATE",
        "SALES_MARKET",
        "SEGMENTS",
        "INTINERARY",
        "BOOKING_ORIGIN_AIRPORT",
        "BOOKING_ORIGIN_COUNTRY_CODE",
        "BOOKING_DEPARTURE_TIME_UTC",
        "BOOKING_DESTINATION_AIRPORT",
        "BOOKING_DESTINATION_COUNTRY_CODE",
        "BOOKING_ARRIVAL_TIME_UTC",
    ]
        
    train_drop_cols = ["UPGRADE_TYPE"]
    df = data.copy()
    if model_type == 'predict_upgrade': 
        train_drop_cols.append("UPGRADE_SALES_DATE")
    elif model_type == 'predict_when_upgrade': 
        train_drop_cols.append("UPGRADED_FLAG")
        df["UPGRADE_SALES_DATE"] = pd.to_datetime(df["UPGRADE_SALES_DATE"])
        df["BOOKING_DEPARTURE_TIME_UTC"] = pd.to_datetime(df["BOOKING_DEPARTURE_TIME_UTC"])

        df["purchase_time_diff"] = df["UPGRADE_SALES_DATE"] - df["BOOKING_DEPARTURE_TIME_UTC"]
        df["purchase_time_diff"] = df["purchase_time_diff"].apply(lambda x: x.days)
        drop_cols.append('UPGRADE_SALES_DATE')

    sus_air_type = ["763", "788", "789", "332", "787"]
    sus_currency = ["JPY", "USD", "CAD", "SGD", "VND", "AED"]
    sus_payment = [
        "UNION",
        "CCDS6",
        "CCSW9",
        "NET R",
        "CCJC3",
        "CCAX3",
        "BARTE",
        "CCVI4",
        "PAY24",
    ]
    df = df[df['TOTAL_PRICE_PLN'] > 0]
    df = df[df['BOOKING_WINDOW_D'] != -1]
    df = df[df['FLIGHT_DISTANCE'] >= 0]
    df["FLIGHT_DATE_LOCAL"] = pd.to_datetime(df["FLIGHT_DATE_LOCAL"])
    df["SALES_DATE"] = pd.to_datetime(df["SALES_DATE"])
    df["sale_to_flight_time"] = df["FLIGHT_DATE_LOCAL"] - df["SALES_DATE"]
    df["sale_to_flight_time"] = df["sale_to_flight_time"].apply(lambda x: x.days)

    stay_lenght_map = {-9999: np.nan}
    df["STAY_LENGTH_D"] = df["STAY_LENGTH_D"].replace(stay_lenght_map)
    df["BOOKING_ARRIVAL_TIME_UTC"] = pd.to_datetime(df["BOOKING_ARRIVAL_TIME_UTC"])
    df["BOOKING_DEPARTURE_TIME_UTC"] = pd.to_datetime(df["BOOKING_DEPARTURE_TIME_UTC"])

    df["flight_len"] = df["BOOKING_ARRIVAL_TIME_UTC"] - df["BOOKING_DEPARTURE_TIME_UTC"]
    df["flight_len"] = df["flight_len"].apply(lambda x: x.seconds / 3600)

    df["TIME_DEPARTURE_LOCAL_TIME"] = pd.to_datetime(df["TIME_DEPARTURE_LOCAL_TIME"])
    df["TIME_DEPARTURE_LOCAL_TIME"] = df["TIME_DEPARTURE_LOCAL_TIME"].apply(
        lambda x: x.hour
    )

    def get_if_add_upgrade(df, data_type):
        emd = pd.read_csv(data_path / f"EMD_{data_type}.csv", sep=";")
        return np.where(
            np.isin(df["TICKET_NUMBER"], emd["REFERENCE_TICKET_NUMBER"].unique()), 1, 0
        )

    df["if_additional_upgrade"] = get_if_add_upgrade(df, data_type=data_type)
    df["same_carrier"] = np.where(
        df["MARKETING_CARRIER"] == df["OPERATIONAL_CARRIER"], 1, 0
    )
    df["is_sus_aircraft"] = np.where(np.isin(df["AIRCRAFT_TYPE"], sus_air_type), 1, 0)

    if data_type == 'train': 
        df["UPGRADED_FLAG"] = df["UPGRADED_FLAG"].map({"Y": 1, "N": 0})

    df["is_sus_payment"] = np.where(np.isin(df["FORM_OF_PAYMENT"], sus_payment), 1, 0)
    df["intinerary_len"] = df["INTINERARY"].apply(lambda x: len(x.split("-")))
    df["is_sus_currency"] = np.where(np.isin(df["FORM_OF_PAYMENT"], sus_currency), 1, 0)

    df["PAX_GENDER"] = df["PAX_GENDER"].map({"M": 1, "F": 0})

    df["CORPORATE_CONTRACT_FLG"] = df["CORPORATE_CONTRACT_FLG"].map({"Y": 1, "N": 0})
    df["LOYAL_CUSTOMER"] = df["LOYAL_CUSTOMER"].map({"Y": 1, "N": 0})

    df["BOOKING_LONG_HOUL_FLAG"] = df["BOOKING_LONG_HOUL_FLAG"].map({"Y": 1, "N": 0})
    df["BOOKING_DOMESTIC_FLAG"] = df["BOOKING_DOMESTIC_FLAG"].map({"Y": 1, "N": 0})

    if data_type == 'train': 
        df = df.drop(columns=train_drop_cols)

    return df.drop(columns=drop_cols)


def oh_encoding(data): 
    df = data.copy()
    oh_cols = [
        "FLIGHT_RANGE",
        "BOOKED_CABIN",
        "VAB",
        "PAX_TYPE",
        "SALES_CHANNEL",
        "TRIP_TYPE",
    ]
    oh_df = pd.get_dummies(data[oh_cols], prefix=oh_cols, drop_first=True)
    return pd.concat([df, oh_df], axis=1).drop(columns=oh_cols)


def data_pipeline(data_path, model_type, data_type): 
    df = read_all_csv(data_path=data_path, data_type=data_type, model_type=model_type)
    clean_data = clean_df(data=df, data_path=data_path, data_type=data_type, model_type=model_type)
    return oh_encoding(clean_data)
