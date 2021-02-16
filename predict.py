if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    
    # reads the data and converts date in the right format
    df = pd.read_csv('/home/dq/scripts/sphist.csv')
    df["Date"] = df["Date"].astype('datetime64[ns]')

    df.sort_values("Date",inplace=True)
    df.reset_index(inplace=True)
    
    
    # creates moving averages

        ## Intuitive computation but super slow
    # df["Avg_5"] = pd.Series([0]*len(df))
    # df["Avg_30"]= pd.Series([0]*len(df))
    # df["Avg_365"] = pd.Series([0]*len(df))
    
    # for i, row in df.iterrows():
    #     if (row["Date"] > datetime(year=1950, month=1, day=9)):
    #         df.loc[i,"Avg_5"] = df.loc[i-5:i]["Close"].mean()
    #     if (row["Date"] > datetime(year=1950, month=2, day=2)):
    #         df.loc[i,"Avg_30"] = df.loc[i-21:i]["Close"].mean()
    #     if (row["Date"] > datetime(year=1951, month=1, day=2)):
    #         df.loc[i,"Avg_365"] = df.loc[i-250:i]["Close"].mean()

        ## Faster computation
    df["Avg_5"] = df["Close"].rolling(5).mean().shift(periods=1)
    df["Avg_30"] = df["Close"].rolling(21).mean().shift(periods=1)
    df["Avg_365"] = df["Close"].rolling(250).mean().shift(periods=1)
    
    # various new features
    df["ratio_5_365"] = df["Avg_5"]/df["Avg_365"]
    df["Std_5"] = df["Close"].rolling(5).std().shift(periods=1)
    df["Avg_Vol_5"] = df["Volume"].rolling(5).std().shift(periods=1)
    df["Avg_Vol_365"] = df["Volume"].rolling(250).std().shift(periods=1)
    df["month"] = df["Date"].dt.month
    dummies = pd.get_dummies(df["month"])
    df = pd.concat([df,dummies],axis=1)
    
    # drops rows before 01/03/51
    df = df[df["Date"] > datetime(year=1951, month=1, day=2)]
    
    # drops NaN values
    df = df.dropna(axis=0)
    
    # creates train and test sets
    train = df[df["Date"] < datetime(year=2013, month=1,day=1)]
    test = df[df["Date"] >= datetime(year=2013, month=1,day=1)]
              
    # instantiates a model
    lr = LinearRegression()
    
    # trains the model
    months = [i for i in range(1,13)]
    columns = ["Avg_5","Avg_30","Avg_365","ratio_5_365", "Std_5","Avg_Vol_365"] + months
    lr.fit(train[columns],train["Close"])
    
    # makes predictions
    predictions = lr.predict(test[columns])
    mae = mean_absolute_error(test["Close"],predictions)
    
    print(mae)
            
            
        