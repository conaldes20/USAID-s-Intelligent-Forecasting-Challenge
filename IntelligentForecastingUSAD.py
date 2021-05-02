import warnings
import itertools
import numpy as np
import xlrd
import csv
import math
import os
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
from datetime import datetime
import matplotlib
import operator
from operator import itemgetter
from scipy import linalg
from pmdarima.arima import ADFTest

def site_product_codes():
    # load the dataset from the CSV file
    reader = csv.reader(open("C:/Users/CONALDES/Documents/IntelligentForecastingSM/contraceptive_logistics_data.csv", "r"), delimiter=",")
    xx = list(reader)
    xxln = len(xx)
    allrecs = []
    for row in range(1, xxln):
        fields = []
        recln = len(xx[row])
        for i in range(0, recln):
            fields.append(xx[row][i])    
        allrecs.append(fields)

    x = np.array(allrecs)
    sitecodes1 = set(x[:,4])
    productcodes1 = set(x[:,5])

    reader = csv.reader(open("C:/Users/CONALDES/Documents/IntelligentForecastingSM/submission_format - submission_format.csv", "r"), delimiter=",")
    xx = list(reader)
    xxln = len(xx)
    allrecs = []
    for row in range(1, xxln):
        fields = []
        recln = len(xx[row])
        for i in range(0, recln):
            fields.append(xx[row][i])    
        allrecs.append(fields)

    x = np.array(allrecs)
    sitecodes2 = set(x[:,2])
    productcodes2 = set(x[:,3])
    
    sitecodes11 = sitecodes1|sitecodes2
    #sitecodes11 = sitecodes|sitecodes3

    productcodes11 = productcodes1|productcodes2
    #productcodes11 = productcodes|productcodes3

    sitecodes11 = list(sitecodes11)
    productcodes11 = list(productcodes11)

    site_codes11_num = len(sitecodes11)
    product_codes11_num = len(productcodes11)
    
    print('sitecodes(unique): ')
    print(sitecodes11)
    print('                            ') 
    print('productcodes(unique) : ')
    print(productcodes11)
    print('                            ') 
    print('site_codes_num, product_codes_num: ', site_codes11_num, product_codes11_num)
    
    return sitecodes11, productcodes11

def createDataTables():
    sitecodes, productcodes = site_product_codes()
    site_codes_num = len(sitecodes)
    product_codes_num = len(productcodes)
    list_of_lists = np.empty((site_codes_num*product_codes_num, 0)).tolist()
    
    # load the dataset from the CSV file 
    reader = csv.reader(open("C:/Users/CONALDES/Documents/IntelligentForecastingSM/contraceptive_logistics_data.csv", "r"), delimiter=",")
    xx = list(reader)
    xxln = len(xx)
    sl_with_max_recno = 0
    lstln = 0
    codes_comb_not_found = []
    for i in range(product_codes_num):
        for j in range(site_codes_num):            
            allrecs = []
            for row in range(1, xxln):
                fields = []
                recln = len(xx[row])                
                for k in range(0, recln):
                    #if k != 2 and k != 3:
                    if k == 0 or k == 1 or k == 4 or k == 5 or k == 8:
                        if k == 0:
                            fields.append(int(xx[row][k]))
                        elif k == 1 or k == 8:
                            fields.append(int(xx[row][k]))
                        else:    
                            fields.append(xx[row][k])
                if productcodes[i].strip() == xx[row][5].strip() and sitecodes[j].strip() == xx[row][4].strip():                    
                    print(fields)
                    allrecs.append(fields)

            if len(allrecs) == 0:
                codes_comb_not_found.append([lstln, sitecodes[j].strip(), productcodes[i].strip()])
                
            if len(allrecs) > sl_with_max_recno:
                sl_with_max_recno = lstln
                
            list_of_lists[lstln].append(allrecs)
            lstln = lstln + 1
            
    years_months = []
    sublist_whnrecs = list_of_lists[sl_with_max_recno][0]
    max_recln = len(sublist_whnrecs)
    for i in range(0, max_recln):  
        years_months.append([sublist_whnrecs[i][0], sublist_whnrecs[i][1]])
        
        
    print('                            ') 
    print('list_of_lists (all tables): ')  
    print(list_of_lists)  
    print('                            ') 
    print('list_of_lists[0][0] (table 1): ')
    print(list_of_lists[0][0])
    print('                            ') 
    print('list_of_lists[1][0] (table 2): ')
    print(list_of_lists[1][0])
    print('                            ') 
    print('information on empty lists: ')
    print(codes_comb_not_found)
    print('                            ')
    print('sublist with max nos of recs: ')
    print(sublist_whnrecs)
    print('                            ')
    print('years and months in data: ')
    print(years_months)
    print('                            ') 
    print('number of recs in sublist_whnrecs: ', len(sublist_whnrecs))
    print('                            ')
    print('number of recs in years_months: ', len(years_months))
    print('                            ')
    print('number of inner lists: ', len(list_of_lists))
    print('                            ')    
    print('number of inner lists without contents: ', len(codes_comb_not_found))
    print('                            ') 
    print('number of inner lists with contents: ', (len(list_of_lists) - len(codes_comb_not_found)))
    print('                            ') 
    
    return list_of_lists, codes_comb_not_found, years_months

       
if __name__ == '__main__':
    now0 = datetime.now()
    timestamp0 = datetime.timestamp(now0)
    
    generated_Ids = []
    predicted_targets_ft = []    
    USAID_list = []    
    
    holtrend_Ids = []
    predicted_targets_htd = []    
    USAID_list_htd = []
    
    esavglst_Ids = []
    predicted_targets_esa = []    
    USAID_list_esa = []
    
    list_of_lists, codes_comb_not_found, years_months = createDataTables()
    lstOflistsln = len(list_of_lists)
    cdcombnotfoundln = len(codes_comb_not_found)
    print('                                             ')
    print("&&&&&&&&&&&&&&&&& Output Details &&&&&&&&&&&&&&&&&")
    print('                                              ')
    
    yearsMonths = np.vstack(years_months)
    
    for iln in range(0, lstOflistsln): 
        temptbl = []
        temptbl0 = list_of_lists[iln][0]        
        if len(temptbl0) > 0:            
            #temptbl.sort(key=itemgetter(0), reverse=True) highest to lowest
            #temptbl.sort(key=itemgetter(0, 1))
            #temptbl = np.vstack(temptb0)
            #print("sorted sublist(temptbl): ")   
            #print(temptbl)   # All fields converted to string
            #print("                                    ")
            
            tempnewtbl = []
            yrmthln = len(yearsMonths)
            temptblln = len(temptbl0)
            sitecod = temptbl0[0][2]
            prodcod = temptbl0[0][3]
            for i in range(0, yrmthln):
                year1 = yearsMonths[i][0]
                month1 = yearsMonths[i][1]  
                found = False                
                for j in range(0, temptblln):
                    year2 = temptbl0[j][0]
                    month2 = temptbl0[j][1]                   
                    if year1 == year2 and month1 == month2:
                        found = True
                        break
                        
                if found == False:
                    temptbl.append([year1, month1, sitecod, prodcod, 0])
            
            for k in range(0, temptblln):
                year = temptbl0[k][0]
                month = temptbl0[k][1] 
                stkdist = temptbl0[k][4]
                temptbl.append([year, month, sitecod, prodcod, stkdist])                
                
            temptbl.sort(key=itemgetter(0, 1))
            temptbl = np.vstack(temptbl) 
            temptblnum = len(temptbl)
            for k in range(0, temptblnum):
                tempnewtbl.append(int(temptbl[k][4]))
            
            #print("sorted sublist(temptbl): ")   
            #print(temptbl)   # All fields converted to string
            #print("                                    ")
            
            data = pd.DataFrame(tempnewtbl, columns =['Target'])
            #print('data: ')
            #print(data)
            #print('                                           ')
            targets = data['Target']
            tgtln = len(targets)
            
            pred_targets_us = []
            pred_targets_ht = []
            pred_targets_es = []    
            holttrd = []
            esavlst = []
            
            for jj in range(0, tgtln):
                intval = int(targets[jj])
                pred_targets_us.append(intval)
                pred_targets_ht.append(intval)
                pred_targets_es.append(intval)
                holttrd.append(intval)
                esavlst.append(intval)
                
            sitecode = temptbl[0][2]
            productcode = temptbl[0][3] 
            
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # Simple Exponential Smoothing    
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            #print('                                                        ')
            #print('@@@@@@@@@@@ Simple Exponential Smoothing @@@@@@@@@@@')
            #print('                                                        ')
            esavgseries = []
            try:
                t = 1
                alpha = 0.4
                yp = []
                for t in range(t, 48):
                    ysum = 0
                    for k in range(0, (t - 1)):            
                        ysum += math.pow((1 - alpha),k)*esavlst[t - k]
                    ysum += math.pow((1 - alpha),t)* esavlst[0]
                    esavlst.append(ysum)
                    val = math.ceil(ysum)
                    if val < 0:
                        val = 0
                    yp.append(val)
                
                esAVGSeries = yp[-3:] 
            
                esavgserlen = len(esAVGSeries)
                for j in range(0, esavgserlen):
                    stdist = math.ceil(esAVGSeries[j])
                    if stdist < 0:
                        stdist = 0
                    esavgseries.append(stdist)
                    pred_targets_es.append(stdist) 
                        
                temprec = [2019, 10, sitecode, productcode, esavgseries[0]]
                USAID_list_esa.append(temprec)                
                temprec = [2019, 11, sitecode, productcode, esavgseries[1]]
                USAID_list_esa.append(temprec)                
                temprec = [2019, 12, sitecode, productcode, esavgseries[2]]
                USAID_list_esa.append(temprec)
            except:
                #print("Exception occured...")
                
                for j in range(0, 3):
                    esavgseries.append(0)
                    pred_targets_es.append(0)
                    
                temprec = [2019, 10, sitecode, productcode, esavgseries[0]]
                USAID_list_esa.append(temprec)                
                temprec = [2019, 11, sitecode, productcode, esavgseries[1]]
                USAID_list_esa.append(temprec)                
                temprec = [2019, 12, sitecode, productcode, esavgseries[2]]
                USAID_list_esa.append(temprec)
            
            usaid_data = pd.DataFrame(pred_targets_es, columns =['Target'])
            actual_usaid_data_es = usaid_data[:46]
            forecast_usaid_data_es = usaid_data[45:]   # Last 6 values
            
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # Holt’s Linear Trend    
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            #print('                                                        ')
            #print('@@@@@@@@@@@ Holt’s Linear Trend @@@@@@@@@@@')
            #print('                                                        ')
            holttrdseries = []
            try:
                alpha = 0.4
                beta = 0.7
                u = []
                u.append(holttrd[0])
                u.append(holttrd[1])
                v = []
                v.append(0)
                v.append(0)    
                yp = []
                yp.append(0)
                for i in range(1, 48):
                    u.append(holttrd[i])
                    templ = alpha*u[i] + (1 - alpha)*(u[i - 1] + v[i - 1])
                    u.append(math.ceil(templ))
                    tempb = beta*(u[i] - u[i - 1]) + (1 - beta)*v[i - 1]
                    v.append(math.ceil(tempb))
                    val = math.ceil(u[i] + v[i])
                    if val < 0:
                        val = 0
                    holttrd.append(val)
                    yp.append(val)
                    
                holtTrdSeries = yp[-3:]    
            
                holttrdserlen = len(holtTrdSeries)
                for j in range(0, holttrdserlen):
                    stdist = math.ceil(holtTrdSeries[j])
                    if stdist < 0:
                        stdist = 0
                    holttrdseries.append(stdist)
                    pred_targets_ht.append(stdist)
                        
                temprec = [2019, 10, sitecode, productcode, holttrdseries[0]]
                USAID_list_htd.append(temprec)                
                temprec = [2019, 11, sitecode, productcode, holttrdseries[1]]
                USAID_list_htd.append(temprec)                
                temprec = [2019, 12, sitecode, productcode, holttrdseries[2]]
                USAID_list_htd.append(temprec)             
            except:
                #print("Exception occured...")
                
                for j in range(0, 3):
                    holttrdseries.append(0)
                    pred_targets_ht.append(0)
                    
                temprec = [2019, 10, sitecode, productcode, holttrdseries[0]]
                USAID_list_htd.append(temprec)                
                temprec = [2019, 11, sitecode, productcode, holttrdseries[1]]
                USAID_list_htd.append(temprec)                
                temprec = [2019, 12, sitecode, productcode, holttrdseries[2]]
                USAID_list_htd.append(temprec) 
            
            usaid_data = pd.DataFrame(pred_targets_ht, columns =['Target'])
            actual_usaid_data_ht = usaid_data[:46]
            forecast_usaid_data_ht = usaid_data[45:]   # Last 6 values            
                          
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # Autoregressive Integrated Moving Average (ARIMA)
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@             
            #print('                                                        ')
            #print('@@@@@@@@@@@ Autoregressive Integrated Moving Average (ARIMA) @@@@@@@@@@@')
            #print('                                                        ')
            fittedseries = []               
            try:
                # Building ARIMA model
                smodel = pm.auto_arima(data, start_p=1, start_q=1,
                                 test='adf',
                                 max_p=3, max_q=3, m=12,
                                 start_P=0, seasonal=True,
                                 d=None, D=1, trace=True,
                                 error_action='ignore',  
                                 suppress_warnings=True, 
                                 stepwise=True)
                smodel.summary()        
        
                # Forecast      (Let’s forecast for the next 6 months. )
                n_periods = 3
                fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
                index_of_fc = pd.date_range(data.index[-1], periods = n_periods, freq='MS')

                # make series for plotting purpose
                fitted_series = pd.Series(fitted, index=index_of_fc)
                lower_series = pd.Series(confint[:, 0], index=index_of_fc)
                upper_series = pd.Series(confint[:, 1], index=index_of_fc)
                              
                fittedserlen = len(fitted_series)
                for j in range(0, fittedserlen):
                    fittedSeries = math.ceil(fitted_series[j])
                    if fittedSeries < 0:
                        fittedSeries = 0
                    fittedseries.append(fittedSeries)
                    pred_targets_us.append(fittedSeries)  
                                
                temprec = [2019, 10, sitecode, productcode, fittedseries[0]]
                USAID_list.append(temprec)                
                temprec = [2019, 11, sitecode, productcode, fittedseries[1]]
                USAID_list.append(temprec)                
                temprec = [2019, 12, sitecode, productcode, fittedseries[2]]
                USAID_list.append(temprec)                 
                
            except:
                #print("Exception occured...")
                
                for j in range(0, 3):
                    fittedseries.append(0)
                    pred_targets_us.append(0)
                                
                temprec = [2019, 10, sitecode, productcode, fittedseries[0]]
                USAID_list.append(temprec)                
                temprec = [2019, 11, sitecode, productcode, fittedseries[1]]
                USAID_list.append(temprec)                
                temprec = [2019, 12, sitecode, productcode, fittedseries[2]]
                USAID_list.append(temprec)            
            
            usaid_data = pd.DataFrame(pred_targets_us, columns =['Target'])
            actual_usaid_data_ar = usaid_data[:46]
            forecast_usaid_data_ar = usaid_data[45:]   # Last 3 values
            
            print('                                             ')
            print("&&&&&&&&&&&&&&&&& Sub-List " + str(iln) + " Summary &&&&&&&&&&&&&&&&&")
            print("                                             ")
            print("sorted sublist: ")   
            temptbl0.sort(key=itemgetter(0, 1))
            temptbl0ln = len(temptbl0)
            for j in range(0, temptbl0ln):
                print(temptbl0[j])
            
            print('                                              ')            
            print('actual stocks distributed for 45 months (missing values replaced with 0): ')
            print(pred_targets_us[:45])
            print('                                              ')
            print('stocks distributed -> Exponential Smoothing Forecast: ')
            print(pred_targets_es[45:])
            print('                                              ')
            print('stocks distributed -> Holt’s Linear Trend Forecast: ')
            print(pred_targets_ht[45:])
            print('                                              ')
            print('stocks distributed -> ARIMA Forecast: ')
            print(pred_targets_us[45:])
            print('                                              ')
            '''            
            try:
                plt.plot(actual_usaid_data_es, label='actual_ES')
                plt.plot(forecast_usaid_data_es, label='forecast_ES')
                plt.plot(actual_usaid_data_ht, label='actual_HT')
                plt.plot(forecast_usaid_data_ht, label='forecast_HT')
                plt.plot(actual_usaid_data_ar, label='actual_AR')
                plt.plot(forecast_usaid_data_ar, label='forecast_AR')
                plt.grid(True)
                plt.legend(loc='best')
                plt.xlabel('month')
                plt.ylabel('stock distributed')
                plt.title('Exponential Smoothing(ES), Holt’s Linear Trend(HT) and ARIMA')
                plt.show()
            except ValueError as err:
                print('Handling run-time error:', err)  
            '''    
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                  
        else:
            for j in range(0, cdcombnotfoundln):
                if iln == codes_comb_not_found[j][0]:                
                    sitecode = codes_comb_not_found[j][1]
                    productcode = codes_comb_not_found[j][2]
                                       
                    temprec = [2019, 10, sitecode, productcode, 0]
                    USAID_list.append(temprec)                    
                    temprec = [2019, 11, sitecode, productcode, 0]
                    USAID_list.append(temprec)
                    temprec = [2019, 12, sitecode, productcode, 0]
                    USAID_list.append(temprec)  
                    
                    temprec = [2019, 10, sitecode, productcode, 0]
                    USAID_list_esa.append(temprec)                
                    temprec = [2019, 11, sitecode, productcode, 0]
                    USAID_list_esa.append(temprec)                
                    temprec = [2019, 12, sitecode, productcode, 0]
                    USAID_list_esa.append(temprec) 
                    
                    temprec = [2019, 10, sitecode, productcode, 0]
                    USAID_list_htd.append(temprec)                
                    temprec = [2019, 11, sitecode, productcode, 0]
                    USAID_list_htd.append(temprec)                
                    temprec = [2019, 12, sitecode, productcode, 0]
                    USAID_list_htd.append(temprec)
           
    USAID_pred_stock_dist_ar = np.vstack(USAID_list)
    USAID_pred_stock_dist_ht = np.vstack(USAID_list_htd) 
    USAID_pred_stock_dist_es = np.vstack(USAID_list_esa)
    
    with open("C:/Users/CONALDES/Documents/IntelligentForecastingSM/ConaldesSubmissionARM_Model.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['year', 'month', 'site_code', 'product_code', 'predicted_value']) 
        for row in USAID_pred_stock_dist_ar:    
            l = list(row)    
            writer.writerow(l)

    print("                                          ")
    print("### C:/Users/CONALDES/Documents/IntelligentForecastingSM/ConaldesSubmissionARM_Model.csv contains results ###")

    with open("C:/Users/CONALDES/Documents/IntelligentForecastingSM/ConaldesSubmissionHLT_Model.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['year', 'month', 'site_code', 'product_code', 'predicted_value']) 
        for row in USAID_pred_stock_dist_ht:    
            l = list(row)    
            writer.writerow(l)

    print("                                          ")
    print("### C:/Users/CONALDES/Documents/IntelligentForecastingSM/ConaldesSubmissionHLT_Model.csv contains results ###")
    
    with open("C:/Users/CONALDES/Documents/IntelligentForecastingSM/ConaldesSubmissionESA_Model.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['year', 'month', 'site_code', 'product_code', 'predicted_value']) 
        for row in USAID_pred_stock_dist_es:    
            l = list(row)    
            writer.writerow(l)

    print("                                          ")
    print("### C:/Users/CONALDES/Documents/IntelligentForecastingSM/ConaldesSubmissionESA_Model.csv contains results ###")
    print("                                          ")       
    now1 = datetime.now()
    timestamp1 = datetime.timestamp(now1)
    time_elapsed = timestamp1 - timestamp0
    print('Time elapsed for computations: ' + str(round(time_elapsed, 2)) + 'seconds')
    
