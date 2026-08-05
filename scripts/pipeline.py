import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer, StandardScaler

def data_engineering(file, fit_spline=None):
    # 13 rows are skipped because the NASA POWER API's first 
    # 13 rows are metadata about the dataset therefore skipped
    if type(file) == str:
        df = pd.read_csv(file, skiprows=13) 
                                            
        # Renamed to simpler names with their units
        df = df.rename(columns={'ALLSKY_SFC_SW_DWN' : 'GHI(W/m2)' , 'WSC' : 'Windspeed(m/s)'}) 
        
        # Renaming to datetime object instead of using 3 diff rows
        time_column = df[['YEAR' , 'MO' , 'DY' , 'HR']].rename(columns={'YEAR' : 'year' , 'MO' : 'month' , 'DY' : 'day' , 'HR' : 'hour'}) 
        df['Time'] = pd.to_datetime(time_column)
        df = df.set_index('Time')
        df = df.drop(columns=['YEAR' , 'MO' , 'DY' , 'HR'])
    else:
        df = file.copy().set_index('Timestamp')
    
    # Hybrid Plant Power Calculation
    
    # Plant Capacities
    sol_AC_cap_MWh = 20
    sol_DC_cap_MWh = 25
    wind_cap_MWh = 10
    battery_cap_MWh = 25
    
    sol_PR = 0.78 # Performance Ratio for Solar
    sol_DC_output = sol_DC_cap_MWh * (df['GHI(W/m2)'] / 1000) * sol_PR
    df['Solar Energy(MWh)'] = sol_DC_output.clip(upper=sol_AC_cap_MWh) # Inverter clipping
    
    velo = df['Windspeed(m/s)']
    
    # Defining boundaries for calculation of wind energy
    wind_energy_rules = [(velo < 3), (velo >= 3) & (velo < 12),    
                         (12 <= velo) & (velo <= 25), (velo > 25)] 
    # below 3(m/s) very slow and above 25 dangerous therefore at those they are fixed to 0
    wind_energy_results = [0 , wind_cap_MWh * ((velo / 12) ** 3), 
                           wind_cap_MWh , 0]                      
    
    df['Wind Energy(MWh)'] = np.select(wind_energy_rules, wind_energy_results, default=0)
    
    df['Total Energy(MWh)'] = df['Solar Energy(MWh)'] + df['Wind Energy(MWh)']
    
    ## Battery charging and storing Logic

    electrolyzer_MW = 9.5
    # Below 40% electrolyzers can be hazardous due to gas mixing
    electrolyzer_safe_MW = 0.4 * electrolyzer_MW 

    total_energy = df['Total Energy(MWh)'].to_numpy()
    battery_flow = np.zeros(len(df))
    cumu = np.zeros(len(df))

    for i in range(len(df)):
        energy = total_energy[i]
        p_cumu = cumu[i]
    
        charge = 0
        discharge = 0
        if energy < electrolyzer_safe_MW: # Low power redirect it to battery
            gap = electrolyzer_safe_MW - energy
        
            if p_cumu < gap:
                charge = energy
            else:
                discharge = gap
        # High power which exceeds max electrolyzer cap and check leftover energy in battery    
        elif energy >  electrolyzer_MW: 
            excess = energy - electrolyzer_MW
            left = battery_cap_MWh - p_cumu

            charge = min(left, excess)

        battery_flow[i] = charge - discharge
        if i < len(df) - 1:
            cumu[i + 1] = min(battery_cap_MWh, max(0, p_cumu + charge - discharge))
    
    df['Battery Charge(MWh)'] = np.where(battery_flow > 0, battery_flow, 0)
    df['Battery Discharge(MWh)'] = np.where(battery_flow < 0, abs(battery_flow), 0)
    df['Stored Energy(MWh)'] = cumu

    energy_to_grid = df['Total Energy(MWh)'] + df['Battery Discharge(MWh)'] - df['Battery Charge(MWh)']
    df['Energy To Grid(MWh)'] = np.where(df['Total Energy(MWh)'] > 9.5 , 9.5 ,  energy_to_grid)

    # Energy consumed by alkaline stack to produce 1kg of H2 due to real world limitations it is 53.2 from 50
    specific_stack_energy_consumption_kWh_per_kg = 53.2 

    # Target Variable 
    df['Hydrogen_yield(kg)'] = df['Energy To Grid(MWh)'] * 1000 / specific_stack_energy_consumption_kWh_per_kg
    
    df['Hr'] = df.index.hour
    df['Mon'] = df.index.month
    df['Day'] = df.index.dayofweek
    
    if fit_spline is None:
        # Using periodic for wrap around after 23:00 and 0:00
        spline = SplineTransformer(n_knots=5, degree=3, extrapolation='periodic') 
        spline_feat = spline.fit_transform(df[['Hr']])
    else:
        # Runs on testing data
        spline = fit_spline
        spline_feat = spline.transform(df[['Hr']])
    
    for i in range(spline_feat.shape[1]):
        col_name = f"spline_hr_{i + 1}"
        df[col_name] = spline_feat[:, i]
    
    # 3hr trends to track sudden changes
    df['Windspeed_mean_3h'] = df['Windspeed(m/s)'].rolling(window=3, min_periods=1).mean()
    df['Windspeed_std_3h'] = df['Windspeed(m/s)'].rolling(window=3, min_periods=1).std().fillna(0)
    df['GHI_mean_3h'] = df['GHI(W/m2)'].rolling(window=3, min_periods=1).mean()
    
    # 1 hr lag features to understand previous state
    
    df['Windspeed_lag_1hr'] = df['Windspeed(m/s)'].shift(1)
    df['GHI_lag_1hr'] = df['GHI(W/m2)'].shift(1)
    
    lags = ['Windspeed_lag_1hr' , 'GHI_lag_1hr']
    df[lags] = df[lags].bfill() # backfill the first row with a valid value

    return df, spline

def classification_engineering(file, scaler = None, spline = None):
    df, spline = data_engineering(file, fit_spline = spline)
    # Safe Threshold is above 40% of Electrolyzer max energy which is 9.5 MWh
    # resulting in 40% of 9.5 = 3.8MWh
    df['Shutdown Threshold'] = (df['Energy To Grid(MWh)'] < 3.8).astype(int)

    features = ['GHI(W/m2)', 'Windspeed(m/s)', 'Stored Energy(MWh)', 'Mon',
                'Day', 'spline_hr_1','spline_hr_2', 'spline_hr_3',
                'spline_hr_4', 'Windspeed_mean_3h','Windspeed_std_3h',
                'GHI_mean_3h', 'Windspeed_lag_1hr','GHI_lag_1hr'
                ]
    X = df[features]
    y = df['Shutdown Threshold']

    if scaler is None:
        # Runs when data is training set and not yet scaled
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    else:
        # When dataset is for testing purpose uses the scaled data from the training phase
        X_scaled = pd.DataFrame(scaler.transform(X), columns=features)

    return X_scaled, y, scaler, spline
