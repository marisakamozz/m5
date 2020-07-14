import logging
import pathlib
import math
import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util import init, reduce_mem_usage


def dump(df, name):
    df = reduce_mem_usage(df)
    save_dir = pathlib.Path('../data/02_fe')
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    joblib.dump(df, save_dir / f'{name}.joblib', compress=True)


def main(run_name):
    calendar = joblib.load('../data/01_readcsv/calendar.joblib')
    prices = joblib.load('../data/01_readcsv/prices.joblib')
    items = joblib.load('../data/01_readcsv/items.joblib')

    prices['diff_price'] = prices.groupby(['store_id', 'item_id'])['sell_price'].transform(pd.Series.diff)
    prices['price_max'] = prices.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
    prices['price_min'] = prices.groupby(['store_id', 'item_id'])['sell_price'].transform('min')
    prices['price_std'] = prices.groupby(['store_id', 'item_id'])['sell_price'].transform('std')
    prices['price_mean'] = prices.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')

    def trend(sell_price):
        sell_price = sell_price.reset_index(drop=True)
        weeks = pd.Series(list(range(len(sell_price))))
        correlation = sell_price.corr(weeks)
        if np.isnan(correlation):
            correlation = 0
        return correlation
    prices['price_trend'] = prices.groupby(['store_id', 'item_id'])['sell_price'].transform(trend)

    prices['price_norm'] = prices['sell_price'] / prices['price_max']
    prices['diff_price_norm'] = prices['diff_price'] / prices['price_max']
    prices['price_nunique'] = prices.groupby(['store_id', 'item_id'])['sell_price'].transform('nunique')

    prices = prices.merge(items)
    prices['dept_max'] = prices.groupby(['store_id', 'dept_id'])['sell_price'].transform('max')
    prices['dept_min'] = prices.groupby(['store_id', 'dept_id'])['sell_price'].transform('min')
    prices['dept_std'] = prices.groupby(['store_id', 'dept_id'])['sell_price'].transform('std')
    prices['dept_mean'] = prices.groupby(['store_id', 'dept_id'])['sell_price'].transform('mean')
    prices['price_in_dept'] = prices['sell_price'] / prices['dept_mean']
    prices['mean_in_dept'] = prices['price_mean'] / prices['dept_mean']
    prices['cat_max'] = prices.groupby(['store_id', 'cat_id'])['sell_price'].transform('max')
    prices['cat_min'] = prices.groupby(['store_id', 'cat_id'])['sell_price'].transform('min')
    prices['cat_std'] = prices.groupby(['store_id', 'cat_id'])['sell_price'].transform('std')
    prices['cat_mean'] = prices.groupby(['store_id', 'cat_id'])['sell_price'].transform('mean')
    prices['price_in_cat'] = prices['sell_price'] / prices['cat_mean']
    prices['mean_in_cat'] = prices['price_mean'] / prices['cat_mean']

    prices = prices.merge(calendar[['wm_yr_wk', 'month', 'year']].drop_duplicates(subset=['wm_yr_wk']))
    prices['price_in_month'] = prices['sell_price'] / prices.groupby(['store_id', 'item_id', 'month'])['sell_price'].transform('mean')
    prices['price_in_year'] = prices['sell_price'] / prices.groupby(['store_id', 'item_id', 'year'])['sell_price'].transform('mean')
    del prices['month'], prices['year']

    dump(prices, 'prices')

    calendar['day'] = calendar['date'].dt.day
    calendar['week'] = calendar['date'].dt.week
    calendar['year_delta'] = (calendar['year'] - calendar['year'].min())
    calendar['week_of_month'] = calendar['day'].apply(lambda x: math.ceil(x/7))
    calendar['day_of_week'] = calendar['date'].dt.dayofweek
    calendar['weekend'] = (calendar['day_of_week']>=5).astype(int)

    holiday = pd.read_csv('../input/holiday/usholidays.csv', index_col=0, parse_dates=['Date'])
    holiday = calendar.merge(holiday, left_on='date', right_on='Date')
    transfer_holidays = holiday[holiday['event_name_1'].isnull()]['date']
    for day in transfer_holidays:
        calendar['event_name_1'].mask(calendar['date'] == day, 'Transfer Holiday', inplace=True)
        calendar['event_type_1'].mask(calendar['date'] == day, 'National', inplace=True)
    calendar['holiday'] = ((calendar['weekend'] == 1) | (calendar['event_type_1'] == 'National')).astype(int)
    calendar['holiday_in_weekday'] = ((calendar['weekend'] == 0) & (calendar['event_type_1'] == 'National')).astype(int)
    
    # sales = pd.read_csv('../input/m5-forecasting-uncertainty/sales_train_validation.csv')
    sales = pd.read_csv('../input/m5-forecasting-uncertainty/sales_train_evaluation.csv')
    sales_columns = [f'd_{i+1}' for i in range(1913)]
    total_sales = sales[sales_columns].sum(axis=0)
    christmas_days = total_sales[total_sales < 10000].index.values
    calendar['christmas_day'] = calendar['d'].apply(lambda x: x in christmas_days).astype(int)

    snap = pd.melt(
        calendar[['d', 'snap_CA', 'snap_TX', 'snap_WI']],
        id_vars='d',
        var_name='state_id',
        value_name='snap'
    )
    snap['state_id'] = snap['state_id'].str[5:]
    calendar.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis=1, inplace=True)

    dump(calendar, 'calendar')
    dump(snap, 'snap')

    v_sales = joblib.load('../data/01_readcsv/v_sales.joblib')
    v_sales = v_sales.merge(calendar)
    v_sales = v_sales.merge(snap)
    v_sales = v_sales.merge(prices)
    cat_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    label_encoders = {}
    for column in tqdm(cat_columns):
        encoder = LabelEncoder()
        v_sales[column] = encoder.fit_transform(v_sales[column].fillna('NA'))
        label_encoders[column] = encoder
    dump(v_sales, 'merge')


if __name__ == "__main__":
    run_name = init(__file__)
    try:
        main(run_name)
    except:
        logging.exception('exception')
    finally:
        logging.info('end')
