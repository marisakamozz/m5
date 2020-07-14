import logging
import pathlib
import joblib
from tqdm import tqdm
import pandas as pd

from util import init, reduce_mem_usage, AGGREGATION_LEVELS


def dump(df, name):
    df = reduce_mem_usage(df)
    save_dir = pathlib.Path('../data/04_agg')
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    joblib.dump(df, save_dir / f'{name}.joblib', compress=True)


def main(run_name):
    v_sales = joblib.load('../data/01_readcsv/v_sales.joblib')
    calendar = joblib.load('../data/02_fe/calendar.joblib')
    prices = joblib.load('../data/02_fe/prices.joblib')
    snap = joblib.load('../data/02_fe/snap.joblib')
    v_sales = v_sales.merge(calendar)
    v_sales = v_sales.merge(snap)
    v_sales = v_sales.merge(prices)

    aggregation_functions = [
        {},
        {},
        {'state_id': 'first'},
        {},
        {'cat_id': 'first'},
        {},
        {'cat_id': 'first'},
        {'state_id': 'first'},
        {'state_id': 'first', 'cat_id': 'first'},
        {'cat_id': 'first', 'dept_id': 'first'},
        {'cat_id': 'first', 'dept_id': 'first'},
        {}
    ]
    common_functions = {
        'sales': 'sum',
        'release_ago': 'mean',
        'wm_yr_wk': 'first',
        'weekday': 'first',
        'wday': 'first',
        'month': 'first',
        'year': 'first',
        'event_name_1': 'first',
        'event_type_1': 'first',
        'event_name_2': 'first',
        'event_type_2': 'first',
        'snap': 'first',
        'day': 'first',
        'week': 'first',
        'year_delta': 'first',
        'week_of_month': 'first',
        'day_of_week': 'first',
        'weekend': 'first',
        'holiday': 'first',
        'holiday_in_weekday': 'first',
        'christmas_day': 'first',
        'sell_price': 'mean',
        'diff_price': 'mean',
        'price_max': 'mean',
        'price_min': 'mean',
        'price_std': 'mean',
        'price_mean': 'mean',
        'price_trend': 'mean',
        'price_norm': 'mean',
        'diff_price_norm': 'mean',
        'price_nunique': 'mean',
        'dept_max': 'mean',
        'dept_min': 'mean',
        'dept_std': 'mean',
        'dept_mean': 'mean',
        'price_in_dept': 'mean',
        'mean_in_dept': 'mean',
        'cat_max': 'mean',
        'cat_min': 'mean',
        'cat_std': 'mean',
        'cat_mean': 'mean',
        'price_in_cat': 'mean',
        'mean_in_cat': 'mean',
        'price_in_month': 'mean',
        'price_in_year': 'mean',
    }
    for a in aggregation_functions:
        a.update(common_functions)
    
    aggregated_dfs = []
    for i, level in tqdm(enumerate(AGGREGATION_LEVELS), total=len(AGGREGATION_LEVELS)):
        logging.info(f'aggregate level = {i}: {level}')
        if i == 11:
            # no aggregation
            df_agg = v_sales.copy()
        else:
            df_agg = v_sales.groupby(level + ['d']).agg(aggregation_functions[i]).reset_index()
            df_agg['sort_key'] = df_agg['d'].str[2:].astype(int)
            df_agg = df_agg.sort_values(level + ['sort_key']).reset_index(drop=True)
            if i == 0:
                df_agg.insert(0, 'id', 'Total_X_evaluation')
            elif len(level) == 1:
                id1 = level[0]
                df_agg.insert(0, 'id', (df_agg[id1] + '_X_evaluation'))
            else:
                id1, id2 = level[0], level[1]
                df_agg.insert(0, 'id', (df_agg[id1] + '_' + df_agg[id2] + '_evaluation'))
        df_agg.insert(0, 'aggregation_level', i)
        aggregated_dfs.append(df_agg)
    
    dump(pd.concat(aggregated_dfs[:9]), 'v_sales_agg')
    dump(pd.concat(aggregated_dfs[9:12]), 'v_sales_each')


if __name__ == "__main__":
    run_name = init(__file__)
    try:
        main(run_name)
    except:
        logging.exception('exception')
    finally:
        logging.info('end')
