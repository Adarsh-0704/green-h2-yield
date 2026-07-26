# Green-Hydrogen-Yield

A Machine Learning project to calculate and optimize green hydrogen yield using NASA POWER Meteorological Data.

## System Boundaries
1) Electrolyzer - Alkaline
2) Capacity - 10,000 KW or 10 MW
3) System Constants - 78% Solar Performance Ratio (PR) which accounts for dust/high temp losses and a 5% (9500 KW) load deduction from original capacity of electrolyzer to cover losses.
4) Minimum Safe Turndown Threshold - 50% (4750 KW) after the deducted capcity.


## Current Project Plan:
1) Load the hourly data of 2014-2024(10 years) from NASA POWER of the solar power (Global Horizontal Irradiance) and wind power.
2) Clean the extracted data.
3) Build sim logic for hybrid solar and wind generation.
4) Train a ML pipeline for real time tracking of operations and forecasting the risks.