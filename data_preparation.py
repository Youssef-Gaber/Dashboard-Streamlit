from configparser import Interpolation
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.signal import medfilt
from TimeSeriesInterpolation import TimeSeriesOOP


def interpolation_subplot(initdf, dataframe, column, method):
    error = 0
    plt.rcParams.update({'xtick.bottom': False})
    #fig, axes = plt.subplots(1, 1, sharex=True, figsize=(20, 20))
    fig = plt.figure(figsize=(10, 4))
    if method=='cubic_fill' or method=='linear_fill':
        #initdf[column].plot(title='Actual', ax=axes[0], label='Actual', color='green', style=".-")
        dataframe[method].plot(title="{} (MSE: ".format(method) + str(error) + ")",  label='{}'.format(method),
                                  color='deeppink',
                                   style=".-")
        st.pyplot(fig)                                                  
    else:
        #initdf[column].plot(title='Actual', ax=axes[0],  label='Actual', color='green', style=".-")
        dataframe[column].plot(title="{} (MSE: ".format(method) + str(error) + ")", label='{}'.format(method),
                                  color='deeppink',
                                   style=".-")
        st.pyplot(fig)        

def linePlot_Out_recogn(dataframe, column):
    fig = plt.figure(figsize=(20, 20))
    sns.lineplot(y = column, x = [i for i in range(len(dataframe[column]))], data = dataframe)
    st.pyplot(fig)

def doubleLinePlot(initdf, dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(y = column, x = [i for i in range(len(initdf[column]))], data = initdf)
    sns.lineplot(y = column, x = [i for i in range(len(dataframe[column]))], data = dataframe)
    st.pyplot(fig)

def BoxPlot(dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.boxplot(y = dataframe[column], orient='v')
    st.pyplot(fig)

def DoubleBoxPlot(initdf, dataframe, column):
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,4))
    sns.boxplot(initdf[column], ax=axes[0], color='skyblue', orient="v")
    axes[0].set_title("Original dataframe")
    sns.boxplot(dataframe[column], ax=axes[1], color='green', orient="v")
    axes[1].set_title("Resulting dataframe")
    fig.tight_layout()
    st.pyplot(fig)

def Histogram(dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data=dataframe, x=column)
    st.pyplot(fig)

def ScatterPlot(initdf, dataframe, column1, column2):
    fig = plt.figure(figsize=(10, 4))
    sns.scatterplot(data=initdf, x=column1, y=column2)
    sns.scatterplot(data=dataframe, x=column1, y=column2)
    st.pyplot(fig)

def removeOutlier(df, columnName, n):
    mean = df[columnName].mean()
    std = df[columnName].std()  
    fromVal = mean - n * std 
    toVal = mean + n * std 
    filtered = df[(df[columnName] >= fromVal) & (df[columnName] <= toVal)] #apply the filtering formula to the column
    return filtered

def removeOutlier_q(df, columnName, n1, n2):
    lower_quantile, upper_quantile = df[columnName].quantile([n1, n2]) #quantiles are generally expressed as a fraction (from 0 to 1)
    filtered = df[(df[columnName] > lower_quantile) & (df[columnName] < upper_quantile)]
    return filtered

def removeOutlier_z(df, columnName, n):
    z = np.abs(stats.zscore(df[columnName])) #find the Z-score for the column
    filtered = df[(z < n)] #apply the filtering formula to the column
    return filtered #return the filtered dataset

def moving_average(dataframe, column, filter_length):
    df_prep = dataframe.copy()
    tair_moving_average = np.convolve(dataframe[column], np.ones((filter_length)), mode ="same")
    tair_moving_average /= filter_length
    tair_moving_average= tair_moving_average[round(filter_length/2): round(len(dataframe[column])-filter_length/2)]
    df_prep[column] = tair_moving_average
    return df_prep

def median_filter(dataframe, column, filter_length):
    """
    data: function that will be filtered
    filter_length: length of the window
    """
    s = dataframe.copy()
    # medfilt_tair = medfilt(dataframe[column], filter_lenght)
    # filtered = dataframe[(dataframe[column] == medfilt_tair)] 
    # # s = pd.DataFrame(medfilt_tair)
    # # s.columns=[column]
    # return filtered

    medfilt_tair = medfilt(dataframe[column], filter_length)
    s[column] = medfilt_tair
    #s = pd.DataFrame(medfilt_tair)
    #s.columns=[column]
    return s


def data_preparation_run(data_obj):
    st.header("DATA PREPARATION")

    if st.sidebar.button("Reset dataframe to the initial one"):
        data_obj.df.to_csv("Prepared Dataset.csv", index=False)

    if pd.read_csv('Prepared Dataset.csv').shape[0] < data_obj.df.shape[0]:
        current_df = pd.read_csv('Prepared Dataset.csv', index_col = None)
    else:
        current_df = data_obj.df

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Original dataframe')
        st.dataframe(data_obj.df)
        st.write(data_obj.df.shape)

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>', unsafe_allow_html=True)
    
    dp_method = st.radio(label = 'Data Prep Method', options = ['Remove outliers','Smoothing','Interpolation'])
    
    if dp_method == 'Remove outliers':
        rmo_radio = st.radio(label = 'Remove outliers method',
                             options = ['Std','Q','Z'])

        if rmo_radio == 'Std':


            with st.container():
                st.subheader('Remove outliers using standard deviation')

                cc1, cc2, cc3, cc4 = st.columns(4)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    std_coeff = st.number_input("Enter standard deviation coefficient (multiplier): ", 0.0, 3.1, 2.0, 0.1)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    rm_outlier = removeOutlier(current_df, selected_column, std_coeff)
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    scatter_plot = st.button('Scatter plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")

                with cc3:
                    scatter_column2 = st.selectbox('Select 2nd column for the scatter plot', [s for s in columns_list if s != selected_column])

                with cc4:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-rm_outlier.shape[0]} rows will be removed.')
                
                if scatter_plot:
                    ScatterPlot(data_obj.df, rm_outlier.reset_index(drop=True), selected_column, scatter_column2)

                if bp:
                    DoubleBoxPlot(data_obj.df, rm_outlier.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(rm_outlier.reset_index(drop=True), selected_column)

                if st.button("Save remove outlier results"):
                    current_df = rm_outlier.reset_index(drop=True)
                    current_df.to_csv("Prepared Dataset.csv", index=False)

        if rmo_radio == 'Q':

            with st.container():
                st.subheader('Remove outliers using quantiles')

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    q_values = st.slider('Select a range of quantiles',
                                        0.00, 1.00, (0.25, 0.75))
                    selected_column = st.selectbox("Select a column:", columns_list)
                    q_outlier = removeOutlier_q(current_df, selected_column, q_values[0], q_values[1])
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-q_outlier.shape[0]} rows will be removed.')
                        
                if bp:
                    DoubleBoxPlot(data_obj.df, q_outlier.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(q_outlier.reset_index(drop=True), selected_column)

                #current_df = rm_outlier.reset_index(drop=True)
                if st.button("Save remove outlier results"):
                    current_df = q_outlier.reset_index(drop=True)
                    current_df.to_csv("Prepared Dataset.csv", index=False)

        if rmo_radio == 'Z':

            with st.container():
                st.subheader('Remove outliers using Z-score')

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    std_coeff = st.number_input("Enter standard deviation coefficient: ", 0.0, 3.1, 2.0, 0.1)
                    #ask Marina about min and max values
                    selected_column = st.selectbox("Select a column:", columns_list)
                    z_outlier = removeOutlier_z(current_df, selected_column, std_coeff)
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-z_outlier.shape[0]} rows will be removed.')
                        
                if bp:
                    DoubleBoxPlot(data_obj.df, z_outlier.reset_index(drop=True), selected_column)

                if hist:
                    Histogram(z_outlier.reset_index(drop=True), selected_column)

                #current_df = rm_outlier.reset_index(drop=True)
                if st.button("Save remove outlier results"):
                    current_df = z_outlier.reset_index(drop=True)
                    current_df.to_csv("Prepared Dataset.csv", index=False)
    if dp_method == 'Smoothing':
        smooth_radio = st.radio(label = 'Smoothing',
                             options = ['Median filter','Moving average','Savitzky Golay'])
        if smooth_radio == 'Median filter':
            
            with st.container():
                st.subheader('Median filter')

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    filter_len = st.slider('Length of the window', 3, 7, 5, 2)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    median_filt = median_filter(current_df, selected_column, filter_len)
                
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                # with cc3:
                #     st.write(" ")
                #     st.write(" ")
                #     st.warning(f'If applied, {current_df.shape[0]-median_filt.shape[0]} rows will be removed.')
                
                if plot_basic:
                    doubleLinePlot(data_obj.df, median_filt.reset_index(drop=True), selected_column)
                    # st.dataframe(median_filt)
                    # st.write(data_obj.df[selected_column].value_counts(ascending=False))
                    # st.write(median_filt[selected_column].value_counts(ascending=False))
                    # st.write("Blah")
                    # l = {'col1': medfilt(data_obj.df[selected_column], filter_len)}
                    # lf = pd.DataFrame(data=l)
                    # st.write(lf.value_counts(ascending=False))


                if bp:
                    DoubleBoxPlot(data_obj.df, median_filt.reset_index(drop=True), selected_column)

                if hist:
                    Histogram(median_filt.reset_index(drop=True), selected_column)

                #current_df = rm_outlier.reset_index(drop=True)
                if st.button("Save remove outlier results"):
                    current_df = median_filt.reset_index(drop=True)
                    current_df.to_csv("Prepared Dataset.csv", index=False)
        if smooth_radio == 'Moving average':
            with st.container():
                st.subheader('Moving average')

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    filter_len = st.slider('Length of the window', 3, 7, 5, 2)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    # median_filt = median_filter(current_df, selected_column, filter_len)
                    moving_ave = moving_average(current_df, selected_column, filter_len)
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-moving_ave.shape[0]} rows will be removed.')
                
                if plot_basic:
                    doubleLinePlot(data_obj.df, moving_ave.reset_index(drop=True), selected_column)
                    # st.dataframe(median_filt)
                    # st.write(data_obj.df[selected_column].value_counts(ascending=False))
                    # st.write(median_filt[selected_column].value_counts(ascending=False))
                    # st.write("Blah")
                    # l = {'col1': medfilt(data_obj.df[selected_column], filter_len)}
                    # lf = pd.DataFrame(data=l)
                    # st.write(lf.value_counts(ascending=False))


                if bp:
                    DoubleBoxPlot(data_obj.df, moving_ave.reset_index(drop=True), selected_column)

                if hist:
                    Histogram(moving_ave.reset_index(drop=True), selected_column)

                #current_df = rm_outlier.reset_index(drop=True)
                if st.button("Save remove outlier results"):
                    current_df = moving_ave.reset_index(drop=True)
                    current_df.to_csv("Prepared Dataset.csv", index=False)

    if dp_method == 'Interpolation':
        interpolation_radio = st.radio(label = 'Interpolation',
                             options = ['Linear','Cubic', 'Forward Fill', 'Backward Fill','All'])
        if interpolation_radio == 'All':
            
            with st.container():
                st.subheader('All interpolations')

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    interpolation_all = TimeSeriesOOP(current_df, selected_column)
                    
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    #bp = st.button("Boxplot")
                    #hist = st.button("Histogram")
                # with cc3:
                #     st.write(" ")
                #     st.write(" ")
                #     st.warning(f'If applied, {current_df.shape[0]-median_filt.shape[0]} rows will be removed.')
                
                if plot_basic:
                    interpolation_all.draw_all(selected_column) 
                    #doubleLinePlot(data_obj.df, interpolation_all.draw_all(selected_column), selected_column)
                    # st.dataframe(median_filt)
                    # st.write(data_obj.df[selected_column].value_counts(ascending=False))
                    # st.write(median_filt[selected_column].value_counts(ascending=False))
                    #st.write("Blah")
                    # l = {'col1': medfilt(data_obj.df[selected_column], filter_len)}
                    # lf = pd.DataFrame(data=l)
                    # st.write(lf.value_counts(ascending=False))


                #if bp:
                    #DoubleBoxPlot(data_obj.df, interpolation_all.draw_all().reset_index(drop=True), selected_column)

                #if hist:
                    #Histogram(interpolation_all.draw_all().reset_index(drop=True), selected_column)

                #current_df = rm_outlier.reset_index(drop=True)
    
        if interpolation_radio == 'Linear':
            
            with st.container():
                st.subheader('Linear interpolation')

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    interpolation_all = TimeSeriesOOP(current_df, selected_column)
                    linear_df = interpolation_all.make_interpolation_liner(selected_column) 
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    #bp = st.button("Boxplot")
                    #hist = st.button("Histogram")
                # with cc3:
                #     st.write(" ")
                #     st.write(" ")
                #     st.warning(f'If applied, {current_df.shape[0]-median_filt.shape[0]} rows will be removed.')
                
                if plot_basic:
                   interpolation_subplot(current_df, linear_df, selected_column, 'linear_fill')
                
                if st.button("Save liner results"):
                    current_df = linear_df.reset_index(drop=True)
                    current_df.to_csv("linear_data.csv", index=False)  
        
        if interpolation_radio == 'Cubic':
            
            with st.container():
                st.subheader('Cubic interpolation')

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    interpolation_all = TimeSeriesOOP(current_df, selected_column)
                    Cubic_df = interpolation_all.make_interpolation_cubic(selected_column) 
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    #bp = st.button("Boxplot")
                    #hist = st.button("Histogram")
                # with cc3:
                #     st.write(" ")
                #     st.write(" ")
                #     st.warning(f'If applied, {current_df.shape[0]-median_filt.shape[0]} rows will be removed.')
                
                if plot_basic: 
                   interpolation_subplot(current_df, Cubic_df, selected_column, 'cubic_fill')
                #    linePlot_Out_recogn(current_df, selected_column)  
                #    linePlot_Out_recogn(Cubic_df, selected_column)
                
                if st.button("Save cubic results"):
                    current_df = Cubic_df.reset_index(drop=True)
                    current_df.to_csv("Cubic_data.csv", index=False)   
        
        if interpolation_radio == 'Forward Fill':
            
            with st.container():
                st.subheader('Forward Fill interpolation')

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    interpolation_all = TimeSeriesOOP(current_df, selected_column)
                    df_ffill = interpolation_all.int_df_ffill() 
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    #bp = st.button("Boxplot")
                    #hist = st.button("Histogram")
                # with cc3:
                #     st.write(" ")
                #     st.write(" ")
                #     st.warning(f'If applied, {current_df.shape[0]-median_filt.shape[0]} rows will be removed.')
                
                if plot_basic:
                   interpolation_subplot(current_df, df_ffill, selected_column, 'Forward Fill')
                
                if st.button("Save Forward Fill results"):
                    current_df = df_ffill.reset_index(drop=True)
                    current_df.to_csv("fforward_data.csv", index=False) 
        
        if interpolation_radio == 'Backward Fill':
            
            with st.container():
                st.subheader('Backward Fill interpolation')

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    interpolation_all = TimeSeriesOOP(current_df, selected_column)
                    df_bfill = interpolation_all.int_df_bfill() 
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    #bp = st.button("Boxplot")
                    #hist = st.button("Histogram")
                # with cc3:
                #     st.write(" ")
                #     st.write(" ")
                #     st.warning(f'If applied, {current_df.shape[0]-median_filt.shape[0]} rows will be removed.')
                
                if plot_basic:
                   #linePlot_Out_recogn(current_df, selected_column) 
                   interpolation_subplot(current_df, df_bfill, selected_column, 'Backward Fill')
                
                if st.button("Save Backward Fill results"):
                    current_df = df_bfill.reset_index(drop=True)
                    current_df.to_csv("backward_data.csv", index=False)                                                
    with col2:
        st.subheader('Current dataframe')
        st.dataframe(current_df) 
        st.write(current_df.shape) 

    with col3:
        st.subheader('Resulting dataframe')
        if dp_method == 'Remove outliers' and rmo_radio == 'Std':
            st.dataframe(rm_outlier.reset_index(drop=True))
            st.write(rm_outlier.shape)
        if dp_method == 'Remove outliers' and rmo_radio == 'Q':
            st.dataframe(q_outlier.reset_index(drop=True))
            st.write(q_outlier.shape)
        if dp_method == 'Remove outliers' and rmo_radio == 'Z':
            st.dataframe(z_outlier.reset_index(drop=True))
            st.write(z_outlier.shape)
        if dp_method == 'Smoothing' and smooth_radio == 'Median filter':
            st.dataframe(median_filt.reset_index(drop=True))
            st.write(median_filt.shape)
        if dp_method == 'Smoothing' and smooth_radio == 'Moving average':
            st.dataframe(moving_ave.reset_index(drop=True))
            st.write(moving_ave.shape)
        if dp_method == 'Interpolation' and  interpolation_radio == 'Linear':
            st.dataframe(linear_df.reset_index(drop=True))
            st.write(linear_df.shape)
        if dp_method == 'Interpolation' and  interpolation_radio == 'Cubic':
            st.dataframe(Cubic_df.reset_index(drop=True))
            st.write(Cubic_df.shape)
        if dp_method == 'Interpolation' and  interpolation_radio == 'Forward Fill':
            st.dataframe(df_ffill.reset_index(drop=True))
            st.write(df_ffill.shape)
        if dp_method == 'Interpolation' and  interpolation_radio == 'Backward Fill':
            st.dataframe(df_bfill.reset_index(drop=True))
            st.write(df_bfill.shape)          
                       
            

