"""
NAME: import_data.py
DESCRIPTION: download data from the mysql database to be run through the dataloader
PROGRAMMER: Caidan Gray
CREATION DATE: 4/3/2025
LAST EDITED: 4/3/2025   (please update each time the script is changed)
"""

import os
import mysql.connector

class Import_Data():
    def __init__(self, training_host: str, training_user: str, training_password: str, testing_host: str, testing_user: str, testing_password: str):
        """initialize needed variables"""
        super(Import_Data, self).__init__()
        #connect to training database
        self.training_db = mysql.connector.connect(
            host=training_host,
            user=training_user,
            password=training_password
        )

        self.training_cursor = self.training_db.cursor()
        #connect to testing database
        self.testing_db = mysql.connector.connect(
            host=testing_host,
            user=testing_user,
            password=testing_password
        )

        self.testing_cursor = self.testing_db.cursor()

    def get_training_data(self, database: str):
        """run a select query to get the training data"""
        self.training_cursor.execute('SELECT * FROM ' + database)  # change SQL query as needed
        train_data = self.training_cursor.fetchall()
        directory = open('Data/Training_Data', 'a')
        directory.write(train_data) # fix input type
        return train_data

    def get_testing_data(self, database: str):
        """run a select query to get the testing data"""
        self.testing_cursor.execute('SELECT * FROM ' + database)  # change SQL query as needed
        test_data = self.testing_cursor.fetchall()
        directory = open('Data/Testing_Data', 'a')
        directory.write(test_data) # fix input type
        return test_data