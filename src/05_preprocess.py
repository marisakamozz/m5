import logging
import pathlib
import joblib
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PowerTransformer

from util import init, reduce_mem_usage, TARGET


def dump(param, dir_name):
    v_sales_dict, data_count, dims, label_encoders, minmax_scalers, power_transformers, sales_transformers = param
    save_dir = pathlib.Path('../data/05_preprocess') / dir_name
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    joblib.dump(v_sales_dict, save_dir / 'v_sales_dict.joblib', compress=True)
    joblib.dump(data_count, save_dir / 'data_count.joblib', compress=True)
    joblib.dump(dims, save_dir / 'dims.joblib', compress=True)
    joblib.dump(label_encoders, save_dir / 'label_encoders.joblib', compress=True)
    joblib.dump(minmax_scalers, save_dir / 'minmax_scalers.joblib', compress=True)
    joblib.dump(power_transformers, save_dir / 'power_transformers.joblib', compress=True)
    joblib.dump(sales_transformers, save_dir / 'sales_transformers.joblib', compress=True)
    del v_sales_dict, data_count, dims, label_encoders, minmax_scalers, power_transformers


def preprocess(v_sales, transform_sales=True):
    logging.info('create dims')
    cat_columns = [
        'aggregation_level',
        'item_id',
        'dept_id',
        'cat_id',
        'store_id',
        'state_id',
        'event_name_1',
        'event_type_1',
        'event_name_2',
        'event_type_2',
    ]
    label_encoders = {}
    remove_columns = []
    for column in tqdm(cat_columns):
        if column in v_sales.columns:
            encoder = LabelEncoder()
            v_sales[column] = encoder.fit_transform(v_sales[column].fillna('NA'))
            label_encoders[column] = encoder
        else:
            remove_columns.append(column)
    for column in remove_columns:
        cat_columns.remove(column)
    v_sales['month'] = v_sales['month'] - 1
    cat_columns = ['day_of_week', 'month'] + cat_columns
    cat_dims = v_sales[cat_columns].nunique()
    dims = pd.DataFrame(cat_dims, columns=['cat_dims'])
    dims['emb_dims'] = cat_dims.apply(lambda x: min(50, (x + 1) // 2))

    minmax_columns = [
        'release_ago',
        'wm_yr_wk',
        'wday',
        'day',
        'week',
        'year_delta',
        'week_of_month',
        'price_nunique',
    ]
    power_columns = [
        'sell_price',
        'diff_price',
        'price_max',
        'price_min',
        'price_std',
        'price_mean',
        # 'price_trend',
        'price_norm',
        'diff_price_norm',
        'dept_max',
        'dept_min',
        'dept_std',
        'dept_mean',
        'cat_max',
        'cat_min',
        'cat_std',
        'cat_mean',
    ]
    logging.info('start MinMaxScaler')
    minmax_scalers = {}
    for column in tqdm(minmax_columns):
        scaler = MinMaxScaler()
        v_sales[column] = scaler.fit_transform(v_sales[[column]])
        minmax_scalers[column] = scaler

    logging.info('start PowerTransformer')
    power_transformers = {}
    for column in tqdm(power_columns):
        logging.info(column)
        scaler = PowerTransformer()
        v_sales[column] = v_sales[column].fillna(0).astype('float64')
        v_sales[column] = scaler.fit_transform(v_sales[[column]])
        v_sales = reduce_mem_usage(v_sales)
        power_transformers[column] = scaler
    
    logging.info('create data_count')
    data_count = pd.DataFrame(v_sales['id'].value_counts()).reset_index()
    data_count.columns = ['id', 'count']

    logging.info('create v_sales_dict and transform sales')
    id_list = v_sales['id'].unique()
    v_sales = v_sales.set_index(['id', 'sort_key']).sort_index()
    v_sales_dict = {}
    sales_transformers = {}
    for data_id in tqdm(id_list):
        data = v_sales.loc[data_id].reset_index()
        data[TARGET] = data[TARGET].astype('float64')  # Int32 -> float64
        if transform_sales:
            scaler = PowerTransformer()
            scaler.fit(data[[TARGET]].iloc[:-28])
            data[TARGET] = scaler.transform(data[[TARGET]])
            sales_transformers[data_id] = scaler
        v_sales_dict[data_id] = data

    return v_sales_dict, data_count, dims, label_encoders, minmax_scalers, power_transformers, sales_transformers


def main(run_name):
    v_sales_agg = joblib.load('../data/04_agg/v_sales_agg.joblib')
    dump(preprocess(v_sales_agg), 'agg_item')
    del v_sales_agg

    v_sales_item = joblib.load('../data/04_agg/v_sales_each.joblib')
    dump(preprocess(v_sales_item, transform_sales=False), 'each_item')
    del v_sales_item


if __name__ == "__main__":
    run_name = init(__file__)
    try:
        main(run_name)
    except:
        logging.exception('exception')
    finally:
        logging.info('end')
