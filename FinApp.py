#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

st.sidebar.header('Financial Funtions Calculations')
# In[5]:
def main():

    finance_functions = ['Homepage', 'Corporate_Finance', 'Portfolio_Management']
    choice=st.sidebar.selectbox('Categories:', finance_functions)

    if choice == 'Homepage':

        st.write("""# **Welcome to Finance Function Calculator Web app** """ )

        st.write("""Explore the different functions at your left navigation sidebar""")

        st.write("""### ** Financial Function for Corporate Finance and Portfolio Management **""")
        st.write("""The Finance Function is a part of financial management. Financial Management is the activity concerned with the control and planning of financial resources.""")

        st.write("""In business, the finance function involves the acquiring and utilization of funds necessary for efficient operations. Finance is the lifeblood of business without it things wouldn’t run smoothly. It is the source to run any organization, it provides the money, it acquires the money.""")

        image = Image.open('Business.jpg')
        st.image(image,use_column_width=True)

        st.balloons()


    elif choice == 'Corporate_Finance':
        activities=['Present_Value','Future_Value','Present_Value_Annuity','Present Value of Annuity Due','Present Value of a Growing Annuity',
        'Gordon Growth Model','Preferred Stock Valuation','Future Value of Annuity Due','Cost of debt',
        'Cost of equity using CAPM method','Cost of Equity by using DDM','Cost of Equity - RPM','Capital Weights','Asset Beta','Degree of Financial Leverage','Degree of Total Leverage_DTL']
        option=st.sidebar.selectbox('Selection option:',activities)

        if option =='Present_Value':
            #st.subheader["Present_Value"]

            image = Image.open('corporate-finance.jpg')
            st.image(image,use_column_width=True)

            st.write("""
            # About Present value """)

            st.write("""
            Present value (PV) is the current value of a future sum of money or stream of cash flows given a specified rate of return. Future cash flows are discounted at the discount rate, and the higher the discount rate, the lower the present value of the future cash flows. Determining the appropriate discount rate is the key to properly valuing future cash flows, whether they be earnings or debt obligations.)
            """)

            st.write("""
            # Present Value Calculator! """)


            def present_value(future_value, discount_rate, periods):

                pv = future_value / ( 1 + discount_rate/100) ** periods
                pre_val = str("Present Value is "+ str("%.3f" % pv))
                st.write(pre_val)
                data_1 = {'Computed_Present_Value' : pre_val}
                result = pd.DataFrame(data_1, index=[0])
                return result


            Period_List = [2,3,4,5,6,7,8,9,10]

            st.sidebar.header('User Input Values')

            def user_input_features():
                future_value = st.sidebar.number_input('Kindly input future value',1000.0)

                discount_rate = st.sidebar.slider('Discount Rate in %', 10,100)

                periods = st.sidebar.selectbox('Select number of periods',Period_List,2)

                data = {'Future_value': future_value,
                        'Discount_rate': discount_rate,
                        'Periods': periods}
                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()
            st.subheader('The calculated present value is as follows below')
            df_1 = present_value(df.Future_value, df.Discount_rate, df.Periods)
            st.write(df_1)

        elif option == 'Present_Value_Annuity':
            st.sidebar.header('User Input Values')

            st.write("""
            # About Present Value of Annuity """)

            st.write("""
            The present value of an annuity is the current value of future payments from an annuity, given a specified rate of return, or discount rate. The higher the discount rate, the lower the present value of the annuity.""")

            image = Image.open('Present-Value-of-Annuity.jpg')
            st.image(image,use_column_width=True)

            st.write("""P = Periodic Payments """)
            st.write("""r = rate per period  """)
            st.write("""n = number of periods """)

            def present_value_annuity(cashflow, discount_rate, periods):
                PVA = cashflow / discount_rate * (1 - 1 / ( 1 + discount_rate/100 )** periods)
                pva_value = str("""### **Present_value_annuity is**: """+str("%.2f" %PVA))
                st.write(pva_value)
                data_4 = {'Computed_present_value_annuity is ': pva_value}
                result = pd.DataFrame(data_4, index=[0])
                return result

            periods_list = [1,2,3,4,5,6,7,8,9,10]
            def user_input_features():
                cashflow = st.sidebar.number_input('Kindly input cashflow')
                discount_rate = st.sidebar.slider('Discount Rate in %', 1,100)
                periods = st.sidebar.selectbox('Select number of periods',periods_list,2)
                data_4 = {'cashflow': cashflow,
                          'discount_rate': discount_rate,
                          'periods': periods}

                features = pd.DataFrame(data_4, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated present value annuity is as follows below')
            df_1 = present_value_annuity(df.cashflow, df.discount_rate, df.periods)
        elif option == 'Present Value of a Growing Annuity':
            st.write("""
            # About Present Value of a Growing Annuity """)

            st.write("""
            ### The present value of a growing annuity is a way to get the current value of a fixed series of cash flows that grow at a proportionate rate. In other words, it is the present value of a series of payments which grows (or declines) at a constant rate each period.
            """)

            image = Image.open('PV_GA.png')
            st.image(image,use_column_width=True)
            def present_value_growing_annuity(cashflow, discount_rate, periods, growth_rate):

                st.write("""
                ### **NB:** The assumption is that the discount rate > growth rate, otherwise a negative value will be returned. """)

                PV_GA = cashflow / discount_rate * (1 - (1 + growth_rate) ** periods / (1 + discount_rate) ** periods)
                pvga_value = str("""## **Present_value_growing_annuity is** """+str("%.2f" %PV_GA))
                st.write(pvga_value)
                data_4 = {'Computed_present_value_growing_annuity is ': pvga_value}
                result = pd.DataFrame(data_4, index=[0])
                return result

            st.sidebar.header('User Input Values')

            periods_list = [1,2,3,4,5,6,7,8,9,10]
            def user_input_features():
                cashflow = st.sidebar.number_input('Kindly input cashflow')
                discount_rate = st.sidebar.slider('Discount Rate in %', 0.01,1.00)
                growth_rate = st.sidebar.slider('Growth Rate in %', 0.01,1.00)
                periods = st.sidebar.selectbox('Select number of periods',periods_list,2)
                data_4 = {'cashflow': cashflow,
                          'discount_rate': discount_rate,
                          'growth_rate': growth_rate,
                          'periods': periods}

                features = pd.DataFrame(data_4, index=[0])
                return features

            df = user_input_features()

            st.subheader("""**The calculated present value growth annuity is as follows below**""")

            df_1 = present_value_growing_annuity(df.cashflow, df.discount_rate, df.periods, df.growth_rate,  )

        elif option == 'Present Value of Annuity Due':
            st.write("""
            # About Present Value of Annuity Due """)

            st.write("""
            The present value of an annuity due (PVAD) is calculating the value at the end of the number of periods given, using the current value of money. Another way to think of it is how much an annuity due would be worth when payments are complete in the future, brought to the present.""")

            image = Image.open('PVA_Due.jpg')
            st.image(image,use_column_width=True)

            st.write("""PMT = Periodic Payment """)
            st.write("""r = rate per period  """)
            st.write("""n = number of periods """)

            def present_value_annuity_due(cashflow, discount_rate, periods):
                PVA_Due = cashflow / discount_rate * (1 - 1 / ( 1 + discount_rate)** periods) * (1 + discount_rate)
                pvad_value = str("""### **Present_value_annuity_due is**: """+str("%.2f" %PVA_Due))
                return pvad_value
            st.sidebar.header('User Input Values')

            periods_list = [1,2,3,4,5,6,7,8,9,10]
            def user_input_features():
                cashflow = st.sidebar.number_input('Kindly input cashflow')
                discount_rate = st.sidebar.slider('Discount Rate in %', 0.01,1.00)
                periods = st.sidebar.selectbox('Select number of periods',periods_list,1)
                data_5 = {'cashflow': cashflow,
                          'discount_rate': discount_rate,
                          'periods': periods}

                features = pd.DataFrame(data_5, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated present value annuity_due is as follows below')
            df_1 = present_value_annuity_due(df.cashflow, df.discount_rate, df.periods)

            st.write(df_1)
        elif option == 'Future_Value':

            image = Image.open('money.jpg')
            st.image(image,use_column_width=True)

            st.write("""
            # About Future value """)

            st.write("""
            Future value (FV) is the value of a current asset at a future date based on an assumed rate of growth. The future value (FV) is important to investors and financial planners as they use it to estimate how much an investment made today will be worth in the future. Knowing the future value enables investors to make sound investment decisions based on their anticipated needs. However, external economic factors, such as inflation, can adversely affect the future value of the asset by eroding its value.""")

            st.write("""
            # Future Value Calculator! """)

            def future_value(present_value, discount_rate, periods):

                fv = present_value * ( 1 + discount_rate/100) ** periods

                pre_val = str("Future Value is "+ str("%.3f" % fv))
                st.write(pre_val)
                data_1 = {'Computed_Future_Value' : pre_val}
                result = pd.DataFrame(data_1, index=[0])
                return result

            Period_List = [2,3,4,5,6,7,8,9,10]

            st.sidebar.header('User Input Values')

            def user_input_features():
                present_value = st.sidebar.number_input('Kindly input present value',1000.0)

                discount_rate = st.sidebar.slider('Discount Rate in %', 10, 100)

                periods = st.sidebar.selectbox('Select number of periods',Period_List,2)

                data = {'Present_value': present_value,
                        'Discount_rate': discount_rate,
                        'Periods': periods}
                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated future value is as follows below: ')
            df_1 = future_value(df.Present_value, df.Discount_rate, df.Periods)

            st.write(df_1)
        elif option == 'Gordon Growth Model':
            st.write("""
            # About Gordon Growth Model (GGM) """)

            st.write("""
            ### The Gordon Growth Model (GGM) is used to determine the intrinsic value of a stock based on a future series of dividends that grow at a constant rate. It is a popular and straightforward variant of the dividend discount model (DDM). The GGM assumes the dividend grows at a constant rate in perpetuity and solves for the present value of the infinite series of future dividends. Because the model assumes a constant growth rate, it is generally only used for companies with stable growth rates in dividends per share.

            """)

            image = Image.open('GGM_Formula.jpg')
            st.image(image,use_column_width=True)

            st.write("""### P = fair value price per share of the equity """)
            st.write("""### D = expected dividend per share one year fro the present time  """)
            st.write("""### g = expected dividend growth rate  """)
            st.write("""### k = required rate of return """)

            def gordon_growth_model(dividend, dividend_growth_rate, required_rate_of_return):

                dividend_period_one = dividend * (1 + dividend_growth_rate)

                gg_m = dividend_period_one / (required_rate_of_return - dividend_growth_rate)
                ggm_value = str("Gordon Growth Model "+str("%.2f" %gg_m))
                st.write(ggm_value)
                data_4 = {'Computed gordon growth model is ': ggm_value}
                result = pd.DataFrame(data_4, index=[0])
                return result


            st.sidebar.header('User Input Values')

            def user_input_features():
                dividend = st.sidebar.number_input('Kindly input dividend')
                growth_rate = st.sidebar.slider('Growth Rate in %', 0.01,1.00)
                rrr = st.sidebar.slider('Select required rate of return', 0.01,1.00)
                data_4 = {'dividend': dividend,
                          'growth_rate': growth_rate,
                          'rrr': rrr}

                features = pd.DataFrame(data_4, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated present value perpetuity is as follows below')

            df_1 = gordon_growth_model(df.dividend, df.growth_rate, df.rrr)

            st.write(df_1)
        elif option == 'Preferred Stock Valuation':
            st.write("""
            # About preferred Stock Valuation """)

            st.write("""
            ### The value of a preferred stock equals the present value of its future dividend payments discounted at the required rate of return of the stock. In most cases the preferred stock is perpetual in nature, hence the price of a share of preferred stock equals the periodic dividend divided by the required rate of return.

            """)
            # In[17]:
            image = Image.open('stock-valuation.jpg')
            st.image(image,use_column_width=True)

            def preferred_stock_valuation(dividend, required_rate_of_return):

                psv = dividend / required_rate_of_return
                psv_value = str("Gordon Growth Model "+str("%.2f" %psv))
                st.write(psv_value)
                data_4 = {'Computed gordon growth model is ': psv_value}
                result = pd.DataFrame(data_4, index=[0])
                return result


            st.sidebar.header('User Input Values')

            def user_input_features():
                dividend = st.sidebar.number_input('Kindly input dividend')
                rrr = st.sidebar.slider('Select required rate of return', 0.01,1.00)
                data_4 = {'dividend': dividend,
                          'rrr': rrr}

                features = pd.DataFrame(data_4, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated present value perpetuity is as follows below')

            df_1 = preferred_stock_valuation(df.dividend,  df.rrr)
        elif option == 'Future Value of Annuity Due':
            st.write("""
            # About Future Value of Annuity Due """)

            st.write("""
            Future value is the value of a sum of cash to be paid on a specific date in the future. An annuity due is a series of payments made at the beginning of each period in the series. Therefore, the formula for the future value of an annuity due refers to the value on a specific future date of a series of periodic payments, where each payment is made at the beginning of a period. Such a stream of payments is a common characteristic of payments made to the beneficiary of a pension plan. These calculations are used by financial institutions to determine the cash flows associated with their products.""")

            image = Image.open('FVA_Due.jpg')
            st.image(image,use_column_width=True)

            st.write("""p = Periodic Payment """)
            st.write("""r = rate per period  """)
            st.write("""n = number of periods """)

            def future_value_annuity_due(cashflow, discount_rate, periods):

                FVA_Due = cashflow / discount_rate * (( 1 + discount_rate )** periods - 1) * (1 + discount_rate)
                fva_value = str(""" ### **Future_value_annuity_due is: ** """+str("%.2f" %FVA_Due))
                st.write(fva_value)
                data_4 = {'Computed_future_value_annuity_due is ': fva_value}
                result = pd.DataFrame(data_4, index=[0])
                return result

            st.sidebar.header('User Input Values')

            periods_list = [1,2,3,4,5,6,7,8,9,10]
            def user_input_features():
                cashflow = st.sidebar.number_input('Kindly input cashflow')
                discount_rate = st.sidebar.slider('Discount Rate in %', 0.01,1.00)
                periods = st.sidebar.selectbox('Select number of periods',periods_list,2)
                data_4 = {'cashflow': cashflow,
                          'discount_rate': discount_rate,
                          'periods': periods}

                features = pd.DataFrame(data_4, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated present value perpetuity is as follows below')

            df_1 = future_value_annuity_due(df.cashflow, df.discount_rate, df.periods)
        elif option == 'Cost of debt':
            st.write("""
            # What is Cost of debt """)

            st.write("""
            The cost of debt is the effective interest rate a company pays on its debts. It’s the cost of debt, such as bonds and loans, among others. The cost of debt often refers to before-tax cost of debt, which is the company's cost of debt before taking taxes into account. However, the difference in the cost of debt before and after taxes lies in the fact that interest expenses are deductible. """)

            image = Image.open('Cost-of-Debt-Formula.jpg')
            st.image(image,use_column_width=True)

            def cost_of_debt(interest_rate, tax_rate):

                cost_of_debt = interest_rate * (1 - tax_rate)
                cps_value = str("### ** Cost of debt ** "+str("%.2f" %cost_of_debt))
                st.write(cps_value)

                return cost_of_debt

            st.sidebar.header('User Input Values')

            def user_input_features():
                interest_rate = st.sidebar.slider('Kindly interest_rate', 0.01,1.00)
                tax_rate = st.sidebar.slider('Kindly tax_rate', 0.01, 1.00)
                data = {'interest_rate': interest_rate,
                        'tax_rate': tax_rate}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated cost of debt is as follows below')

            df_1 = cost_of_debt(df.interest_rate, df.tax_rate)
        elif option == 'Cost of equity using CAPM method':
            st.write(""" ## **What is Cost of equity using CAPM method** """)
            st.write(""" The cost of equity can be calculated by using the CAPM (Capital Asset Pricing Model) CAPM formula shows the return of a security is equal to the risk-free return plus a risk premium, based on the beta of that security or Dividend Capitalization Model (for companies that pay out dividends).""")

            image = Image.open('Cost-of-Equity-Formula.jpg')
            st.image(image,use_column_width=True)

            st.write ("""### **Where**""")
            st.write("""**Risk_free_rate**: The risk-free rate for the market, usually a treasury note.""")
            st.write("""**Market_return:** The required rate of return for the company.""")
            st.write("""**Beta:** The company's estimated stock beta.""")
            st.write("""**CAPM:** Capital Asset Pricing Model (CAPM). """)

            def cost_of_equity_capm(risk_free_rate, market_return, beta):

                cost_of_equity_capm = risk_free_rate + (beta * (market_return - risk_free_rate))
                cec_value = str("**  Cost of equity capm **"+str("%.2f" %cost_of_equity_capm))
                st.write(cec_value)

                return cost_of_equity_capm

            st.sidebar.header('User Input Values')

            def user_input_features():
                risk_free_rate = st.sidebar.slider('Kindly enter risk free rate', 0.01,1.00)
                market_return = st.sidebar.slider('Kindly enter market return', 0.01, 1.00)
                beta = st.sidebar.slider('Kindly enter beta value', 0.01, 1.00)

                data = {'risk_free_rate': risk_free_rate,
                        'market_return': market_return,
                        'beta': beta,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated cost of equity capm is as follows below')

            df_1 = cost_of_equity_capm(df.risk_free_rate, df.market_return, df.beta)
        elif option == 'Cost of Equity by using DDM':
            st.write("""## **What is Cost of Equity by using Dividend Discount Model** """)
            st.write("""Cost of equity can be worked out with the help of Gordon’s Dividend Discount Model. The model focuses on the dividends as the name suggests. According to the model, the cost of equity is a function of current market price and the future expected dividends of the company. The rate at which these two things are equal is the cost of equity.""")

            st.write(""" It is the simple phenomenon of ‘what is the cost of buying equity’ and ‘what will I get from it’. Here, ‘what is the cost of buying equity’ represents the current market price of that equity share and ‘what will I get from it’ is represented by the expected future dividends of the company. By comparing the two, we can get the actual rate of return which an investor will get as per the current situations. That rate of return is the cost of equity. The underlying assumption here is that the current market price is adjusted as per the required rate of return by the investor in that share.""")

            image = Image.open('Cost-of-Equity-DDM.jpg')
            st.image(image,use_column_width=True)

            st.write("""### **Where**""")
            st.write("""* **stock_price:** The company's current price of a share.""")
            st.write("""* **Next_year_dividend:** The expected dividend to be paid next year.""")
            st.write("""* **growth_rate:** Firm's expected constant growth rate.""")

            def cost_of_equity_ddm(stock_price, next_year_dividend, growth_rate):

                cost_of_equity_ddm = (next_year_dividend / stock_price) + growth_rate
                coe_value = str("### **Cost of equity using DDM **"+str("%.2f" %cost_of_equity_ddm))
                st.write(coe_value)

                return coe_value


            st.sidebar.header('User Input Values')

            def user_input_features():
                stock_price = st.sidebar.number_input('Kindly enter stock price' )
                next_yr_dividend = st.sidebar.slider('Kindly enter market return', 1, 100)
                growth_rate = st.sidebar.slider('Kindly select your growth-rate', 0.01, 1.00)

                data = {'stock_price': stock_price,
                        'next_yr_dividend': next_yr_dividend,
                        'growth_rate': growth_rate,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated cost of equity DDM is as follows below')

            df_1 = cost_of_equity_ddm(df.stock_price, df.next_yr_dividend, df.growth_rate)
        elif option == 'Cost of Equity - RPM':
            st.write("""## **What is Cost of Equity - Risk Premium Method** """)

            st.write("""The term equity risk premium refers to an excess return that investing in the stock market provides over a risk-free rate. This excess return compensates investors for taking on the relatively higher risk of equity investing. The size of the premium varies and depends on the level of risk in a particular portfolio. It also changes over time as market risk fluctuates.""")

            image = Image.open('cost_equity_bond.jpg')
            st.image(image,use_column_width=True)

            st.write("""## **Where:**""")
            st.write("""* **DPS** = dividends per share, for next year""")
            st.write("""* **CMV** = current market value of stock""")
            st.write("""* **GRD** = growth rate of dividends""")

            def cost_of_equity_bond(bond_yield, risk_premium):

                ceb = bond_yield + risk_premium

                ceb_value = str("### **Cost of equity bond is** "+str("%.2f" %ceb))
                st.write(ceb_value)

                return ceb_value

            st.sidebar.header('User Input Values')

            def user_input_features():
                bond_yield = st.sidebar.slider('Kindly select bond_yield', 0.01, 1.00)
                risk_premium = st.sidebar.slider('Kindly enter risk_premium', 0.01, 1.00)

                data = {'bond_yield': bond_yield,
                        'risk_premium': risk_premium,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated cost of equity bond is as follows below')

            df_1 = cost_of_equity_bond(df.bond_yield, df.risk_premium)
        elif option == 'Capital Weights':
            st.write("""## **What is Capital Weights** """)

            st.write("""Given a firm's capital structure, calculate the weights of each group.""")

            image = Image.open('capital_weights.jpg')
            st.image(image,use_column_width=True)

            st.write("""
            ### **Total_capital:** The company's total capital.
            ### **Preferred_stock:** The company's preferred stock outstanding.
            ### **common_stock:** The company's common stock outstanding.
            ### **Total_debt:** The company's total debt """)

            def capital_weights(preferred_stock, total_debt, common_stock):

                weights_dict = {}

                total_capital = preferred_stock + common_stock + total_debt

                weights_dict['preferred_stock'] = preferred_stock / total_capital
                weights_dict['common_stock'] = common_stock / total_capital
                weights_dict['total_debt'] = total_debt / total_capital

                st.write(preferred_stock)
                st.write(common_stock)
                st.write(total_debt)

                st.write("""Hence the total capital is""", total_capital)

                cap_weights1 = str("**Capital weights on preferred stock is:** "+str("%.2f" %weights_dict['preferred_stock']))
                cap_weights2 = str("**Capital weights on common stock is:** "+str("%.2f" %weights_dict['common_stock']))
                cap_weights3 = str("**Capital weights on total_debt is:** "+str("%.2f" %weights_dict['total_debt']))

                st.write(cap_weights1)
                st.write(cap_weights2)
                st.write(cap_weights3)

                return weights_dict

            st.sidebar.header('User Input Values')

            def user_input_features():
                preferred_stock = st.sidebar.number_input('Kindly enter preferred stock amount')
                total_debt = st.sidebar.number_input('Kindly enter total debt amount')
                common_stock = st.sidebar.number_input('Kindly enter common stock amount')

                data = {'preferred_stock': preferred_stock,
                        'total_debt': total_debt,
                        'common_stock': common_stock}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated capital weight is as follows below')

            df_1 = capital_weights(df.preferred_stock, df.total_debt, df.common_stock)
        elif option == 'Asset Beta':
            st.write("""## **What is Asset Beta (Unlevered beta)** """)
            st.write("""Beta is a measure of market risk. Unlevered beta (or asset beta) measures the market risk of the company without the impact of debt. Unlevering a beta removes the financial effects of leverage thus isolating the risk due solely to company assets. In other words, how much did the company's equity contribute to its risk profile.""")

            image = Image.open('asset_beta.jpg')
            st.image(image,use_column_width=True)

            st.write("""
            ### **Summary:** Calculate the asset beta for a publicly traded firm.
            >**Tax_rate:** A comparable publicly traded company's marginal tax rate.

            >**Equity_beta:** A comparable publicly traded company's equity beta.

            >**Debt_to_equity:** A comparable publicly traded company's debt-to-equity ratio.
            """)

            def asset_beta(tax_rate, equity_beta, debt_to_equity):

                asset_beta = equity_beta * 1 / (1 + ((1 - tax_rate) * debt_to_equity))

                ab_value = str("### **The Asset Beta is given as:** "+str("%.2f" %asset_beta))
                st.write(ab_value)

                return ab_value

            st.sidebar.header('User Input Values')

            def user_input_features():
                tax_rate = st.sidebar.slider('Select tax rate', 0.01, 1.00)
                equity_beta = st.sidebar.slider('Select equity beta', 0.01, 1.00)
                debt_to_equity = st.sidebar.slider('Select debt to equity', 0.01, 1.00)

                data = {'tax_rate': tax_rate,
                        'equity_beta': equity_beta,
                        'debt_to_equity': debt_to_equity,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated asset beta is as follows below')

            df_1 = asset_beta(df.tax_rate, df.equity_beta, df.debt_to_equity)
        elif option == 'Degree of Financial Leverage':
            st.write("""## **What Is a Degree of Financial Leverage - DFL?**""")

            st.write("""A degree of financial leverage (DFL) is a leverage ratio that measures the sensitivity of a company’s earnings per share (EPS) to fluctuations in its operating income, as a result of changes in its capital structure. The degree of financial leverage (DFL) measures the percentage change in EPS for a unit change in operating income, also known as earnings before interest and taxes (EBIT).""")

            st.write("""This ratio indicates that the higher the degree of financial leverage, the more volatile earnings will be. Since interest is usually a fixed expense, leverage magnifies returns and EPS. This is good when operating income is rising, but it can be a problem when operating income is under pressure.""")

            image = Image.open('Degree-of-Financial-Leverage-Formula.jpg')
            st.image(image,use_column_width=True)

            def degree_of_financial_leverage(ebit, interest):

                dfl = ebit / (ebit - interest)

                dfl_value = str("### **The Degree of financial leverage is given as:** "+str("%.2f" %dfl))
                st.write(dfl_value)
                return dfl_value

            st.sidebar.header('User Input Values')

            def user_input_features():
                ebit = st.sidebar.number_input('Kindly input EBIT')
                interest = st.sidebar.number_input('Kindly input interest')

                data = {'ebit': ebit,
                        'interest': interest,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated degree of financial leverage is as follows below')

            df_1 = degree_of_financial_leverage(df.ebit, df.interest)
        elif option == 'Degree of Total Leverage_DTL':
            st.write("""## **What is a Degree of Total Leverage - DTL?**""")

            st.write("""The degree of total leverage is a ratio that compares the rate of change a company experiences in earnings per share (EPS) to the rate of change it experiences in revenue from sales.""")

            st.write("""The degree of total leverage can also be referred to as the “degree of combined leverage” because it considers the effects of both operating leverage and financial leverage.""")

            image = Image.open('Degree_TL.png')
            st.image(image,use_column_width=True)

            def degree_of_total_leverage(financial_leverage, operating_leverage):

                dtl = financial_leverage * operating_leverage

                dtl_value = str("### **The Degree of total leverage is given as:** "+str("%.2f" %dtl))
                st.write(dtl_value)

                return financial_leverage * operating_leverage

            st.sidebar.header('User Input Values')

            def user_input_features():
                financial_leverage = st.sidebar.slider('Kindly financial leverage', 1.00, 10.00)
                operating_leverage = st.sidebar.slider('Kindly input operating leverage', 1.00, 10.00)

                data = {'financial_leverage': financial_leverage,
                        'operating_leverage': operating_leverage,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated degree of total leverage is as follows below')

            df_1 = degree_of_total_leverage(df.financial_leverage, df.operating_leverage)

    elif choice == 'Portfolio_Management':
        activities=['Holding Period Return/Yield','Expected Return','Capital Market Line','Modigliani risk-adjusted performance',
        'Treynor Measure','Jensens Alpha','Return on Equity','Bond Value','Current Yield','Forward Rate']
        option=st.sidebar.selectbox('Selection option:',activities)

        if option =='Holding Period Return/Yield':
            st.write("""## **What Is the Holding Period Return/Yield?**""")
            st.write("""Holding period return is the total return received from holding an asset or portfolio of assets over a period of time, known as the holding period, generally expressed as a percentage. Holding period return is calculated on the basis of total returns from the asset or portfolio (income plus changes in value). It is particularly useful for comparing returns between investments held for different periods of time.""")

            image = Image.open('Holding-Period-Return-Formula.jpg')
            st.image(image,use_column_width=True)

            st.write("""### **SUMMARY:** Calculate the holding period return for an asset. """)

            st.write(""" > **Price_end (Ending Value):** The price at the end of the period.""")
            st.write(""" > **Price_beginning (Initial Value):** The price at the beginning of the period. """)
            st.write(""" > **Dividends_earned (Income Generated):** The dividends earned during the period""")

            def holding_period_return(price_end, price_beginning, dividends_earned):

                hpr = (price_end + dividends_earned) / price_beginning - 1

                hpr_value = str("### **The holding period return is given as: **"+str("%.2f" %hpr))
                st.write(hpr_value)

                return  hpr_value

            st.sidebar.header('User Input Values')

            def user_input_features():
                price_end = st.sidebar.slider('Kindly select price end',1.00,10.00)
                price_beginning = st.sidebar.slider('Kindly select price beginning',1.00,10.00)
                dividends_earned = st.sidebar.slider('Kindly dividends earned', 1.00, 10.00)

                data = {'price_end': price_end,
                        'price_beginning': price_beginning,
                        'dividends_earned': dividends_earned,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated holding period return is as follows below')

            df_1 = holding_period_return(df.price_end, df.price_beginning, df.dividends_earned)
        elif option =='Expected Return':
            st.write("""## **What is Expected Return?**""")
            st.write("""The expected return is the profit or loss that an investor anticipates on an investment that has known historical rates of return (RoR). It is calculated by multiplying potential outcomes by the chances of them occurring and then totaling these results. Expected return calculations are a key piece of both business operations and financial theory, including in the well-known models of modern portfolio theory (MPT) or the black-scholes options pricing model.""")

            image = Image.open('Expected-Return-Formula.jpg')
            st.image(image,use_column_width=True)

            st.write("""### **Where**""")

            st.write(""" **Ri** – Return Expectation of each scenario""")
            st.write(""" **Pi** – Probability of the return in that scenario""")
            st.write(""" **i** – Possible Scenarios extending from 1 to n""")

            def expected_returns(market_return, beta, risk_free_rate):

                er = risk_free_rate + (beta * (market_return - risk_free_rate))
                er_value = str("### **The expected return is given as:** "+str("%.2f" %er))
                st.write(er_value)

                return er_value

            st.sidebar.header('User Input Values')

            def user_input_features():
                market_return = st.sidebar.slider('Kindly select market return',0.01,1.00)
                beta = st.sidebar.slider('Kindly select beta',1.00,10.00)
                risk_free_rate = st.sidebar.slider('Kindly risk free rate', 0.01, 1.00)

                data = {'market_return': market_return,
                        'beta': beta,
                        'risk_free_rate': risk_free_rate,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated expected return is as follows below')

            df_1 = expected_returns(df.market_return, df.beta, df.risk_free_rate)
        elif option =='Capital Market Line':
            st.write("""## **What is Capital Market Line?**""")

            st.write("""The Capital Market Line is a graphical representation of all the portfolios that optimally combine risk and return. CML is a theoretical concept that gives optimal combinations of a risk-free asset and the market portfolio.""")

            image = Image.open('Capital_ML.jpg')
            st.image(image,use_column_width=True)

            st.write("""### **SUMMARY:** Calculate the capital market line for a portfolio""")

            st.write("""* **Market_return:** The expected return for the market.""")
            st.write("""* **Std_market:** The standard deviation for market returns.""")
            st.write("""* **Risk_free_rate:** The rate of return for  a risk free asset, usually t-bill.""")

            def capital_market_line(return_market, risk_free_rate, std_market):

                cml = (return_market - risk_free_rate) / std_market

                cml_value = str("### **The expected return is given as:** "+str("%.2f" %cml))
                st.write(cml_value)

                return cml_value


            st.sidebar.header('User Input Values')

            def user_input_features():
                return_market = st.sidebar.slider('Kindly select price end',0.01,1.00)
                risk_free_rate = st.sidebar.slider('Kindly enter risk free rate', 0.01, 1.00)
                std_market = st.sidebar.slider('Kindly enter standard deviation for market returns',0.01,1.00)

                data = {'return_market': return_market,
                        'risk_free_rate': risk_free_rate,
                        'std_market': std_market,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated capital market line is as follows below')

            df_1 = capital_market_line(df.return_market, df.risk_free_rate, df.std_market)

        elif option == 'Modigliani risk-adjusted performance':
            st.write("""## **What Is M2**""")

            st.write("""Modigliani risk-adjusted performance (also known as M2, M2, Modigliani–Modigliani measure or RAP) is a measure of the risk-adjusted returns of some investment portfolio. It measures the returns of the portfolio, adjusted for the risk of the portfolio relative to that of some benchmark (e.g., the market). We can interpret the measure as the difference between the scaled excess return of our portfolio P and that of the market, where the scaled portfolio has the same volatility as the market. It is derived from the widely used Sharpe ratio, but it has the significant advantage of being in units of percent return (as opposed to the Sharpe ratio – an abstract, dimensionless ratio of limited utility to most investors), which makes it dramatically more intuitive to interpret.""")

            image = Image.open('M-Squared-Measure.jpg')
            st.image(image,use_column_width=True)

            st.write("""### **SUMMARY:** Calculate the m-squared for a portfolio""")

            st.write("""**Return_portfolio:** The expected return for the portfolio.""")
            st.write("""**Std_portfolio:** The standard deviation for portfolio returns.""")
            st.write("""**Market_return:** The expected return for the market.""")
            st.write("""**Market_std:** The standard deviation for market returns.""")
            st.write("""**Risk_free_rate:** The rate of return for  a risk free asset, usually t-bill.""")

            def m_squared(return_portfolio, risk_free_rate, return_market, std_market, std_portfolio):

                m_squared = ((return_portfolio - risk_free_rate) * (std_market / std_portfolio)) + (return_market - risk_free_rate)
                msquared_value = str("### **The m_squared for a portfolio is given as:** "+str("%.2f" %m_squared))
                st.write(msquared_value)

                return m_squared

            st.sidebar.header('User Input Values')

            def user_input_features():
                return_portfolio = st.sidebar.slider('Kindly select price end',0.01,1.00)
                risk_free_rate = st.sidebar.slider('Kindly enter risk free rate', 0.01, 1.00)
                return_market = st.sidebar.slider('Kindly enter return market', 0.01, 1.00)
                std_market = st.sidebar.slider('Kindly enter standard deviation for market returns',0.01,1.00)
                std_portfolio = st.sidebar.slider('Kindly enter standard portfolio',0.01, 1.00)

                data = {'return_portfolio': return_portfolio,
                        'risk_free_rate': risk_free_rate,
                        'return_market': return_market,
                        'std_market': std_market,
                        'std_portfolio': std_portfolio,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated m_squared for a portfolio is as follows below')

            df_1 = m_squared(df.return_portfolio, df.risk_free_rate, df.return_market, df.std_market, df.std_portfolio)

        elif option == 'Treynor Measure':
            st.write("""## **What is Treynor measure**""")

            st.write("""The Treynor ratio, also known as the reward-to-volatility ratio, is a performance metric for determining how much excess return was generated for each unit of risk taken on by a portfolio.
            Excess return in this sense refers to the return earned above the return that could have been earned in a risk-free investment. Although there is no true risk-free investment, treasury bills are often used to represent the risk-free return in the Treynor ratio.
            Risk in the Treynor ratio refers to systematic risk as measured by a portfolio's beta. Beta measures the tendency of a portfolio's return to change in response to changes in return for the overall market.""")

            image = Image.open('treynor_ratio.png')
            st.image(image,use_column_width=True)

            def treynor_measure(return_portfolio, risk_free_rate, beta_portfolio):

                treynor_measure = (return_portfolio - risk_free_rate) / beta_portfolio

                tm_value = str("### **The treynor measure is given as:** "+str("%.2f" %treynor_measure))
                st.write(tm_value)

                return tm_value

            st.sidebar.header('User Input Values')

            def user_input_features():
                return_portfolio = st.sidebar.slider('Kindly select return portfolio',0.01,1.00)
                risk_free_rate = st.sidebar.slider('Kindly enter risk free rate', 0.01, 1.00)
                beta_portfolio = st.sidebar.slider('Kindly enter beta portfolio', 0.01, 1.00)

                data = {'return_portfolio': return_portfolio,
                        'risk_free_rate': risk_free_rate,
                        'beta_portfolio': beta_portfolio,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated treynor_measure is as follows below')

            df_1 = treynor_measure(df.return_portfolio, df.risk_free_rate, df.beta_portfolio)

        elif option == 'Jensens Alpha':
            st.write("""## **What is Jensens Alpha**""")
            st.write("""Jensen's alpha is a formula used to calculate an investment's risk-adjusted value. Also referred to as Jensen's Performance Index and ex-post alpha, Jensen's alpha aims to determine the abnormal return of a portfolio or security, with 'security' referring to any asset including stocks, bonds and derivatives.""")

            image = Image.open('Jensens-Alpha.png')
            st.image(image,use_column_width=True)

            st.write("""### **SUMMARY:** Calculate the jensen alpha measure for a portfolio""")
            st.write("""* **Return_portfolio:** The expected return for the portfolio.""")
            st.write("""* **Risk_free_rate:** The rate of return for  a risk free asset, usually t-bill.""")
            st.write("""* **Beta_portfolio:** The portfolio beta.""")
            st.write("""* **Market_return:** The expected return for the market.""")

            def jensen_alpha(return_portfolio, risk_free_rate, beta_portfolio, return_market):

                jensen_alpha = return_portfolio - (risk_free_rate + beta_portfolio * (return_market - risk_free_rate))

                ja_value = str("### **The jensens alpha is given as:** "+str("%.2f" %jensen_alpha))
                st.write(ja_value)
                return ja_value

            st.sidebar.header('User Input Values')

            def user_input_features():
                return_portfolio = st.sidebar.slider('Kindly select return portfolio',0.01,1.00)
                risk_free_rate = st.sidebar.slider('Kindly enter risk free rate', 0.01, 1.00)
                beta_portfolio = st.sidebar.slider('Kindly enter beta portfolio', 0.01, 1.00)
                return_market = st.sidebar.slider('Kindly enter return market', 0.01, 1.00)

                data = {'return_portfolio': return_portfolio,
                        'risk_free_rate': risk_free_rate,
                        'beta_portfolio': beta_portfolio,
                        'return_market': return_market,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated jensa alpha is as follows below')

            df_1 = jensen_alpha(df.return_portfolio, df.risk_free_rate, df.beta_portfolio, df.return_market)

        elif option == 'Return on Equity':
            st.write("""## **What is Return on Equity**""")
            st.write("""Return on equity (ROE) is a measure of financial performance calculated by dividing net income by shareholders' equity. Because shareholders' equity is equal to a company’s assets minus its debt, ROE is considered the return on net assets. ROE is considered a measure of the profitability of a corporation in relation to stockholders’ equity.""")

            image = Image.open('ROE.jpg')
            st.image(image,use_column_width=True)

            def return_on_equity(net_income, book_value):

                ROE = net_income / book_value
                ROE_value = str("### **The return on equity is given as:** "+str("%.2f" %ROE))
                st.write(ROE_value)
                return ROE_value

            st.sidebar.header('User Input Values')

            def user_input_features():
                net_income = st.sidebar.number_input('Kindly input net_income')
                book_value = st.sidebar.slider('Kindly enter book value', 1.0, 30.00)

                data = {'net_income': net_income,
                        'book_value': book_value,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated return on equity is as follows below')

            df_1 = return_on_equity(df.net_income, df.book_value)
        elif option == 'Bond Value':
            st.write("""## **What is Bond Valuation**""")

            st.write("""Bond valuation is a technique for determining the theoretical fair value of a particular bond. Bond valuation includes calculating the present value of a bond's future interest payments, also known as its cash flow, and the bond's value upon maturity, also known as its face value or par value. Because a bond's par value and interest payments are fixed, an investor uses bond valuation to determine what rate of return is required for a bond investment to be worthwhile.""")

            image = Image.open('Bond-Formula.jpg')
            st.image(image,use_column_width=True)

            st.write("""### **SUMMARY:** Calculate the value of a bond, given the maturity value, number of years to maturity and interest rate.""")

            st.write("""* **Maturity_value:** The value of the bond at maturity.""")
            st.write("""* **Interest_rate:** The semi-annual discount rate on the bond.""")
            st.write("""* **Num_of_years:** The life of the bond in N years.""")

            def bond_value(maturity_value, interest_rate, num_of_years):

                bond_value = maturity_value / (1 + interest_rate) ** (num_of_years * 2)

                bond_value = str("### **The bond value is given as:** "+str("%.2f" %bond_value))
                st.write(bond_value)

                return bond_value

            st.sidebar.header('User Input Values')

            def user_input_features():
                maturity_value = st.sidebar.number_input('Kindly input net_income')
                interest_rate = st.sidebar.slider('Kindly enter book value', 1.0, 30.00)
                num_of_years = st.sidebar.slider('Select the number of years')

                data = {'maturity_value': maturity_value,
                        'interest_rate': interest_rate,
                        'num_of_years': num_of_years,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated bond value is as follows below')

            df_1 = bond_value(df.maturity_value, df.interest_rate, df.num_of_years)

        elif option == 'Current Yield':
            st.write("""## **What is Current Yield**""")
            st.write("""Current yield is an investment's annual income (interest or dividends) divided by the current price of the security. This measure examines the current price of a bond, rather than looking at its face value. Current yield represents the return an investor would expect to earn, if the owner purchased the bond and held it for a year. However, current yield is not the actual return an investor receives if he holds a bond until maturity.""")

            image = Image.open('Current-Yield-of-Bond.jpg')
            st.image(image,use_column_width=True)

            st.write("""### **SUMMARY:** Calculate the current yield of a bond.""")
            st.write(""" * **Annual_coup_pay:** The coupon payment made annually.""")
            st.write(""" * **bond_price:** the bonds current price.""")

            def current_yield(annual_coup_pay, bond_price):

                current_yield = annual_coup_pay / bond_price

                cy_value = str("### **The current yield is given as:** "+str("%.2f" %current_yield))
                st.write(cy_value)

                return cy_value

            st.sidebar.header('User Input Values')

            def user_input_features():
                annual_coup_pay = st.sidebar.number_input('Kindly enter annual_coup_pay')
                bond_price = st.sidebar.number_input('Kindly enter bond price')

                data = {'annual_coup_pay': annual_coup_pay,
                        'bond_price': bond_price,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated current yield is as follows below')

            df_1 = current_yield(df.annual_coup_pay, df.bond_price)

        elif option == 'Forward Rate':
            st.write("""## **What is Forward rate**""")

            st.write("""A forward rate is an interest rate applicable to a financial transaction that will take place in the future. Forward rates are calculated from the spot rate and are adjusted for the cost of carry to determine the future interest rate that equates the total return of a longer-term investment with a strategy of rolling over a shorter-term investment.
            The term may also refer to the rate fixed for a future financial obligation, such as the interest rate on a loan payment.""")

            image = Image.open('Forward-Rate-Formula.jpg')
            st.image(image,use_column_width=True)

            st.write("""### **SUMMARY:** Calculate the forward rate given a spot rate.""")
            st.write("""**Spot_rate:** The spot rate for each period.""")
            st.write("""**periods:** The number of periods to be calculated. """)

            def forward_rate(spot_rate, periods):

                spot_n = (1 + (spot_rate * periods)) ** periods
                spot_n_1 = (1 + (spot_rate * (periods - 1))) ** (periods - 1)

                forward_rate = (spot_n / spot_n_1) - 1
                fr_value = str("### **The forward rate is given as:** "+str("%.2f"%forward_rate))
                st.write(fr_value)

                return forward_rate

            st.sidebar.header('User Input Values')

            def user_input_features():
                spot_rate = st.sidebar.number_input('Kindly enter spot rate')
                periods = st.sidebar.number_input('Kindly enter number of periods')

                data = {'spot_rate': spot_rate,
                        'periods': periods,}

                features = pd.DataFrame(data, index=[0])
                return features

            df = user_input_features()

            st.subheader('The calculated forward rate is as follows below')

            df_1 = forward_rate(df.spot_rate, df.periods)

if __name__ == '__main__':
    main()


# In[ ]:
