import logging
import pathlib
import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util import INDEX_COLUMNS, init, reduce_mem_usage

def dump(df, name):
    df = reduce_mem_usage(df)
    save_dir = pathlib.Path('../data/01_readcsv')
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    joblib.dump(df, save_dir / f'{name}.joblib', compress=True)


def main(run_name):
    input_dir = pathlib.Path('../input/m5-forecasting-uncertainty')
    calendar = pd.read_csv(input_dir / 'calendar.csv', parse_dates=['date'])
    prices = pd.read_csv(input_dir / 'sell_prices.csv')
    # sales = pd.read_csv(input_dir / 'sales_train_validation.csv')
    sales = pd.read_csv(input_dir / 'sales_train_evaluation.csv')
    items = sales[['item_id', 'dept_id', 'cat_id']].drop_duplicates()

    releases = prices.groupby(['store_id','item_id'])['wm_yr_wk'].min().reset_index()
    releases.columns = ['store_id','item_id','wm_yr_wk']
    weekday = calendar.groupby('wm_yr_wk')['date'].min().reset_index()
    releases = releases.merge(weekday)
    releases.columns = ['store_id','item_id','release_week', 'release_date']
    releases.drop('release_week', axis=1, inplace=True)

    for d in calendar['d']:
        if d not in sales.columns:
            sales[d] = pd.NA
    v_sales = pd.melt(
        sales,
        id_vars=INDEX_COLUMNS,
        var_name='d',
        value_name='sales'
    )
    v_sales['sales'] = v_sales['sales'].astype('Int32')

    v_sales = v_sales.merge(releases)
    v_sales = v_sales.merge(calendar[['d', 'date']])
    v_sales = v_sales[v_sales['date'] >= v_sales['release_date']].copy().reset_index(drop=True)
    v_sales['release_ago'] = (v_sales['date'] - v_sales['release_date']).dt.days

    dump(v_sales, 'v_sales')
    dump(calendar, 'calendar')
    dump(prices, 'prices')
    dump(items, 'items')

    v_sales = v_sales.merge(calendar)
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
