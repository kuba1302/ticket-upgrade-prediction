import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class Pipeline:
    def __init__(
        self,
        data_path: str = Path(os.getcwd()) / "WEC2022_Data",
        data_type: str = "train",
        model_type: str = "predict_upgrade",
    ):
        self.data_path, self.data_type, self.model_type = (
            data_path,
            data_type,
            model_type,
        )
        self.df = self.merge_files()
        self.concat_df_with_oh_encoding()

    def read_csv_file(self, file_prefix) -> pd.DataFrame:
        return pd.read_csv(
            self.data_path / f"{file_prefix}_{self.data_type}.csv", sep=";"
        )

    def read_all_files(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        bkg = (
            self.read_csv_file("BKG").drop(columns=self.get_bkg_drop_cols())
            if self.data_type == "train"
            else self.read_csv_file("BKG")
        )
        logger.info(f"{datetime.now()} loaded booking data")
        tkt = self.read_csv_file("TKT")
        logger.info(f"{datetime.now()} loaded ticket data")
        fcp = self.read_csv_file("FCP")
        logger.info(f"{datetime.now()} loaded flight coupon data")
        return (
            bkg,
            tkt,
            fcp[fcp.UPGRADED_FLAG == "Y"]
            if self.model_type == "predict_when_upgrade"
            else fcp,
        )

    def merge_files(self) -> pd.DataFrame:
        bkg, tkt, fcp = self.read_all_files()
        logger.info(f"{datetime.now()} merging datasets")
        return fcp.merge(tkt, on="TICKET_NUMBER", how="left").merge(
            bkg, on="BOOKING_ID", how="left"
        )

    @staticmethod
    def discard_nonupgraded_rows(fcp: pd.DataFrame) -> pd.DataFrame:
        return fcp[fcp["UPGRADED_FLAG"] == "Y"]

    @staticmethod
    def get_bkg_drop_cols() -> list:
        return ["UPGRADED_FLAG", "UPGRADE_TYPE", "UPGRADE_SALES_DATE"]

    @staticmethod
    def get_columns_to_drop() -> list:
        return [
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

    def get_cols_to_drop_for_training(self) -> list:
        cols = ["UPGRADE_TYPE"]
        cols.append(
            "UPGRADE_SALES_DATE"
        ) if self.model_type == "predict_upgrade" else cols.append("UPGRADED_FLAG")
        return cols

    def get_new_cols_for_predict_when_model(self) -> None:
        self.df["purchase_time_diff"] = (
            self.df["UPGRADE_SALES_DATE"] - self.df["BOOKING_DEPARTURE_TIME_UTC"]
        )
        self.df["purchase_time_diff"] = self.df["purchase_time_diff"].apply(
            lambda x: x.days
        )

    @staticmethod
    def get_sus_air_type() -> list:
        return ["763", "788", "789", "332", "787"]

    @staticmethod
    def get_sus_currency() -> list:
        return ["JPY", "USD", "CAD", "SGD", "VND", "AED"]

    @staticmethod
    def get_sus_payment() -> list:
        return [
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

    def filter_wrong_ticket_prices(self) -> None:
        self.df = self.df[self.df["TOTAL_PRICE_PLN"] > 0]

    @staticmethod
    def get_datetime_columns() -> list:
        return [
            "UPGRADE_SALES_DATE",
            "BOOKING_DEPARTURE_TIME_UTC",
            "FLIGHT_DATE_LOCAL",
            "SALES_DATE",
            "BOOKING_ARRIVAL_TIME_UTC",
            "TIME_DEPARTURE_LOCAL_TIME",
        ]

    def convert_datetime_columns_to_pandas_format(self) -> None:
        self.df[self.get_datetime_columns()] = self.df[
            self.get_datetime_columns()
        ].apply(pd.to_datetime)

    def filter_wrong_booking_window(self) -> None:
        self.df = self.df[self.df["BOOKING_WINDOW_D"] != -1]

    def filter_nonpositive_flight_distance(self) -> None:
        self.df = self.df[self.df["FLIGHT_DISTANCE"] >= 0]

    def get_sale_to_flight_time(self) -> None:
        self.df["sale_to_flight_time"] = (
            self.df["FLIGHT_DATE_LOCAL"] - self.df["SALES_DATE"]
        )
        self.df["sale_to_flight_time"] = self.df["sale_to_flight_time"].apply(
            lambda x: x.days
        )

    @staticmethod
    def get_stay_length_map() -> dict:
        return {-9999: np.nan}

    def map_stay_length(self) -> None:
        self.df["STAY_LENGTH_D"] = self.df["STAY_LENGTH_D"].replace(
            self.get_stay_length_map()
        )

    def get_flight_len(self) -> None:
        self.df["flight_len"] = (
            self.df["BOOKING_ARRIVAL_TIME_UTC"] - self.df["BOOKING_DEPARTURE_TIME_UTC"]
        )
        self.df["flight_len"] = self.df["flight_len"].apply(lambda x: x.seconds / 3600)

    def map_departure_time_to_hours(self) -> None:
        self.df["TIME_DEPARTURE_LOCAL_TIME"] = self.df[
            "TIME_DEPARTURE_LOCAL_TIME"
        ].apply(lambda x: x.hour)

    @staticmethod
    def get_yes_no_binary_map() -> dict:
        return {"Y": 1, "N": 0}

    def check_for_add_upgrade(self) -> None:
        emd = self.read_csv_file("EMD")
        self.df["if_additional_upgrade"] = np.where(
            np.isin(self.df["TICKET_NUMBER"], emd["REFERENCE_TICKET_NUMBER"].unique()),
            1,
            0,
        )

    def check_for_same_carrier(self) -> None:
        self.df["same_carrier"] = np.where(
            self.df["MARKETING_CARRIER"] == self.df["OPERATIONAL_CARRIER"], 1, 0
        )

    def check_for_sus_aircraft(self) -> None:
        self.df["is_sus_aircraft"] = np.where(
            np.isin(self.df["AIRCRAFT_TYPE"], self.get_sus_air_type()), 1, 0
        )

    def map_target_variable_for_training(self) -> None:
        self.df["UPGRADED_FLAG"] = self.df["UPGRADED_FLAG"].map(
            self.get_yes_no_binary_map()
        )

    def check_for_sus_payment(self) -> None:
        self.df["is_sus_payment"] = np.where(
            np.isin(self.df["FORM_OF_PAYMENT"], self.get_sus_payment()), 1, 0
        )

    def get_intinerary_len(self) -> None:
        self.df["intinerary_len"] = self.df["INTINERARY"].apply(
            lambda x: len(x.split("-"))
        )

    def check_for_sus_currency(self) -> None:
        self.df["is_sus_currency"] = np.where(
            np.isin(self.df["FORM_OF_PAYMENT"], self.get_sus_currency()), 1, 0
        )

    @staticmethod
    def get_map_gender() -> dict:
        return {"M": 1, "F": 0}

    def map_genders(self) -> None:
        self.df["PAX_GENDER"] = self.df["PAX_GENDER"].map(self.get_map_gender())

    def map_corporate_contract_flg(self) -> None:
        self.df["CORPORATE_CONTRACT_FLG"] = self.df["CORPORATE_CONTRACT_FLG"].map(
            self.get_yes_no_binary_map()
        )

    def map_loyal_customer(self) -> None:
        self.df["LOYAL_CUSTOMER"] = self.df["LOYAL_CUSTOMER"].map(
            self.get_yes_no_binary_map()
        )

    def map_booking_long_houl_flag(self) -> None:
        self.df["BOOKING_LONG_HOUL_FLAG"] = self.df["BOOKING_LONG_HOUL_FLAG"].map(
            self.get_yes_no_binary_map()
        )

    def map_booking_domestic_flag(self) -> None:
        self.df["BOOKING_DOMESTIC_FLAG"] = self.df["BOOKING_DOMESTIC_FLAG"].map(
            self.get_yes_no_binary_map()
        )

    def clean_df(self):
        logger.info(f"{datetime.now()} cleaning df ...")
        self.convert_datetime_columns_to_pandas_format()
        logger.info(f"{datetime.now()} converted datetime columns to pandas format")
        self.filter_wrong_ticket_prices()
        logger.info(f"{datetime.now()} filtered wrong ticket prices")
        self.filter_wrong_booking_window()
        logger.info(f"{datetime.now()} filtered wrong booking windows")
        self.filter_nonpositive_flight_distance()
        logger.info(f"{datetime.now()} filtered nonpositive flight distance")
        self.get_sale_to_flight_time()
        logger.info(f"{datetime.now()} got sale to flight time col")
        self.get_flight_len()
        logger.info(f"{datetime.now()} got flight len col")
        self.map_departure_time_to_hours()
        logger.info(f"{datetime.now()} mapped departure time to hours")
        self.check_for_add_upgrade()
        logger.info(f"{datetime.now()} checked for add upgrade")
        self.check_for_same_carrier()
        logger.info(f"{datetime.now()} checked for same carrier")
        self.check_for_sus_aircraft()
        logger.info(f"{datetime.now()} checked for sus aircraft")
        self.map_target_variable_for_training() if self.data_type == "train" else None
        self.check_for_sus_payment()
        logger.info(f"{datetime.now()} checked for sus payment")
        self.get_intinerary_len()
        logger.info(f"{datetime.now()} got itinerary len col")
        self.check_for_sus_currency()
        logger.info(f"{datetime.now()} checked for sus currency")
        self.map_genders()
        logger.info(f"{datetime.now()} mapped genders")
        self.map_corporate_contract_flg()
        logger.info(f"{datetime.now()} mapped corporate contract flag")
        self.map_loyal_customer()
        logger.info(f"{datetime.now()} mapped loyal customers")
        self.map_booking_long_houl_flag()
        logger.info(f"{datetime.now()} mapped booking long houl flag")
        self.map_booking_domestic_flag()
        logger.info(f"{datetime.now()} mapped booking domestic flag")
        self.df = (
            self.df.drop(columns=self.get_cols_to_drop_for_training())
            if self.data_type == "train"
            else self.df
        )
        self.df.drop(columns=self.get_columns_to_drop(), inplace=True)
        logger.info(f"{datetime.now()} dropped unnecessary columns")

    @staticmethod
    def get_oh_cols() -> list:
        return [
            "FLIGHT_RANGE",
            "BOOKED_CABIN",
            "VAB",
            "PAX_TYPE",
            "SALES_CHANNEL",
            "TRIP_TYPE",
        ]

    def get_oh_encoding(self) -> pd.DataFrame:
        logger.info(f"{datetime.now()} getting OH encoding")
        return pd.get_dummies(
            self.df[self.get_oh_cols()], prefix=self.get_oh_cols(), drop_first=True
        )

    def concat_df_with_oh_encoding(self) -> None:
        self.clean_df()
        self.df = pd.concat([self.df, self.get_oh_encoding()], axis=1).drop(
            columns=self.get_oh_cols()
        )


if __name__ == "__main__":
    d = Pipeline().df
