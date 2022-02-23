# General import section
import pandas as pd #to work with dataframes
import streamlit as st #streamlit backend
from io import StringIO #to read data files as .csv correctly

# Dashboard apps
from data_preview import data_preview_run #page 1: Data Preview
from smoothing_and_filtering import smoothing_and_filtering_run #page 2: Smoothing and Filtering
from dp_scaler_split import scaler_split_run #page 3: Scaling and Train-test split

# Data object class
class DataObject():
    """
    Data object class holds a dataframe and its byte size.
    """
    def __init__(self, df=None, filesize=None):
      """The constructor for DataObject class.

      Args:
          df (pandas.core.frame.DataFrame, optional): pandas dataframe object. Defaults to None.
          filesize (numpy.int32, optional): byte size of pandas dataframe. Defaults to None.
      """
      self.df = df
      self.filesize = filesize

# Interface class        
class Interface():
    """
    Interface class contains a file picker and a side bar. It also handles the import of a data object.
    """
    def __init__(self):
      """The constructor for Interface class.
      """
      pass
    
    def side_bar(cls, dt_obj):
      """Sidebar configuration and file picker

      Args:
          dt_obj (pandas.core.frame.DataFrame): pandas dataframe object.
      """
      # Accepts .csv and .data
      filename = st.sidebar.file_uploader("Upload a data file", type=(["csv", "data"]))                   
      if filename is not None: #file uploader selected a file      
        try: #most datasets can be read using standard 'read_csv'                                                                                           
            dt_obj.df = pd.read_csv(filename, sep=';|,', decimal=',', engine='python')                                                                     
            dt_obj.filesize = dt_obj.df.size
        except: #due to a different encoding some datafiles require additional processing
            filename.seek(0)
            filename = filename.read()
            filename = str(filename,'utf-8')
            filename = StringIO(filename)
            #now the standard 'read_csv' should work
            dt_obj.df = pd.read_csv(filename, sep=';', decimal=',', index_col = False)
            dt_obj.filesize = dt_obj.df.size


        # Attempts to convert all numerical columns to 'datetime' format. Relevant for timeseries.
        # for col in dt_obj.df.columns:
        #   if dt_obj.df[col].dtype == 'object':
        #     try:
        #       dt_obj.df[col] = pd.to_datetime(dt_obj.df[col])
        #     except ValueError:
        #       pass
      
        # Side bar navigation menu with a select box
        menu = ['Data Preview', 'Smoothing and filtering', 'Data Preparation', 'Classification', 'Regression']
        navigation = st.sidebar.selectbox(label="Select menu", options=menu)

        # Runs 'Data Preview' app
        if navigation == 'Data Preview':
          with st.container():
           data_preview_run(dt_obj)

        # Runs 'Smoothing and filtering' app
        if navigation == 'Smoothing and filtering':
          smoothing_and_filtering_run(dt_obj)

        # Runs 'Data Preparation' app
        if navigation == 'Data Preparation':
          scaler_split_run(dt_obj)
        
        # Runs 'Classification' app
        if navigation == 'Classification':
          st.header("CLASSIFICATION")

        # Runs 'Regression' app
        if navigation == 'Regression':
          st.header('REGRESSION')
         

def main():
  """
  Main and its Streamlit configuration
  """
  st.set_page_config(page_title="MAIT 21/22 Data Analytics Dashboard",
                     page_icon=None,
                     layout="wide",
                     initial_sidebar_state="expanded",
                     menu_items=None)
  # Creating an instance of the original dataframe data object                   
  data_main = DataObject()
  # Creating an instance of the main interface
  interface = Interface()
  interface.side_bar(data_main)


# Run Main
if __name__ == '__main__':
  main()
