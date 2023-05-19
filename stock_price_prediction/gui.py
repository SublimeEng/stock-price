from bsedata.bse import BSE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import our finalise model
import pickle
from bse_scrap import bse_scrape
import streamlit as st

bseScrap = bse_scrape()
Company_list = bseScrap.get_security_code()

st.title("Stock price prediction of BSE")

user_input = st.selectbox("select company", Company_list )
seurity_code = bseScrap.get_securityCode_by_company(user_input)
st.write(seurity_code)
data = bseScrap.get_databysecurity(seurity_code)

st.write(data.describe())
#st.write(type(data))
#plot graph
st.subheader('Closing price vs time')
fig = plt.figure(figsize = (12,6))
plt.plot(data.Date, data.Close,'red')
#plt.show()
st.pyplot(fig)


#plot graph
st.subheader('Closing price vs Ema')
fig = plt.figure(figsize = (12,6))
plt.plot(data.Date, data.ema,'black')
plt.plot(data.Date, data.Close,'red')
#plt.show()
st.pyplot(fig)

r2_score_lst, Linearreg, Lassoreg, Ridgereg, regressor, ls, rr = bseScrap.train_test(data)

r2_score_maxindex = r2_score_lst.index(max(r2_score_lst))
result = r2_score_maxindex + 1
        
if result == 1:
    pickle.dump(regressor, open('Linearmodel.pkl', 'wb'))
    #Linearreg.plot(y=['y_test','y_pred_lr'])
    st.subheader('prediction vs actual')
    fig2 = plt.figure(figsize = (12,6))
    sr = Linearreg.y_test.to_frame()
    
    ss = Linearreg.y_pred_lr.to_frame()
    print("type of linear y_test is " + str(type(sr)))
    print("type of linear y_pred_lr is " + str(type(Linearreg.y_pred_lr)))
    print(sr)
    print(ss)
    sr.reset_index(level=0, drop=True).reset_index()
    ss.reset_index(level=0, drop=True).reset_index()
    ss['date']  = ss.index
    print(ss.date)
    #sr.set_index(pd.DatetimeIndex(sr['Date']), inplace=True)
    #ss.set_index(pd.DatetimeIndex(ss['Date']), inplace=True)
    x= pd.array([48])
    plt.plot(sr)
    plt.plot(ss)
    #plt.plot(Linearreg.y_test,'r', label = 'Original Price')
    #plt.plot(Linearreg.y_pred_lr,'b', label = 'Predicted Price')
    
    
    #print("Actual:" + Linearreg['y_test'], "Predicted:" +Linearreg['y_pred_rr'])

if result == 2:
    Lassoreg.plot(y=['y_test','y_pred_ls'])
    pickle.dump(ls, open('Lassomodel.pkl', 'wb'))

if result == 3:
    Ridgereg.plot(y=['y_test','y_pred_rr'])
    pickle.dump(rr, open('Ridgemodel.pkl', 'wb'))
    

plt.show()
st.pyplot(fig2)

