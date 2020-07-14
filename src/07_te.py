import argparse
import logging
import pathlib
import joblib
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler

from util import init


def parse_args(run_name):
    parser = argparse.ArgumentParser(description=run_name)
    parser.add_argument('--scaler', type=str, default='PowerTransformer')
    return parser.parse_args()


def dump(target_encoding, file_name):
    save_dir = pathlib.Path('../data/07_te')
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    joblib.dump(target_encoding, save_dir / f'{file_name}.joblib', compress=True)


def target_encode(v_sales, calendar, transform_sales=True, scaler=None):
    v_sales = v_sales[['id', 'd', 'sales']].merge(calendar[['d', 'day_of_week']])
    target_encoding = v_sales.groupby(['id', 'day_of_week'])['sales'].mean().reset_index()
    if transform_sales:
        target_encoding = target_encoding.pivot(index='id', columns='day_of_week', values='sales').T.to_dict(orient='list')
        sales_transformers = joblib.load('../data/05_preprocess/agg_item/sales_transformers.joblib')
        for data_id, scaler in sales_transformers.items():
            target_encoding[data_id] = list(scaler.transform(np.array(target_encoding[data_id]).reshape(-1, 1)).reshape(-1))
    else:
        if scaler == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            scaler = PowerTransformer()
        target_encoding['sales'] = scaler.fit_transform(target_encoding[['sales']])
        target_encoding = target_encoding.pivot(index='id', columns='day_of_week', values='sales').T.to_dict(orient='list')
    return target_encoding



def main(run_name):
    args = parse_args(run_name)
    calendar = joblib.load('../data/02_fe/calendar.joblib')
    v_sales_agg = joblib.load('../data/04_agg/v_sales_agg.joblib')
    dump(target_encode(v_sales_agg, calendar), 'agg_te')

    v_sales_item = joblib.load('../data/04_agg/v_sales_each.joblib')
    dump(target_encode(v_sales_item, calendar, transform_sales=False, scaler=args.scaler), 'each_te')


if __name__ == "__main__":
    run_name = init(__file__)
    try:
        main(run_name)
    except:
        logging.exception('exception')
    finally:
        logging.info('end')
