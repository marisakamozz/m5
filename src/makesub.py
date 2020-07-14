import logging
import argparse
import pathlib
import joblib
import pandas as pd

from util import init, AGGREGATION_LEVELS, INDEX_COLUMNS

def parse_args(run_name):
    parser = argparse.ArgumentParser(description=run_name)
    parser.add_argument('agg_datetime', type=str)
    parser.add_argument('each_datetime', type=str)
    return parser.parse_args()

def modify_acc(acc):
    path_eval = pathlib.Path('../data/id_list_eval.joblib')
    path_valid = pathlib.Path('../data/id_list_valid.joblib')
    if path_eval.exists():
        id_list_eval = joblib.load(path_eval)
        id_list_valid = joblib.load(path_valid)
    else:
        v_sales_agg = joblib.load('../data/04_agg/v_sales_agg.joblib')
        v_sales_each = joblib.load('../data/04_agg/v_sales_each.joblib')
        id_list_agg = v_sales_agg[['id', 'aggregation_level', 'state_id', 'store_id', 'cat_id', 'dept_id']].drop_duplicates()
        id_list_each = v_sales_each[['id', 'aggregation_level', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id']].drop_duplicates()
        id_list_eval = pd.concat([id_list_agg, id_list_each]).reset_index(drop=True)
        id_list_eval['id'] = id_list_eval['id'].str[:-10] + 'evaluation'
        id_list_valid = id_list_eval.copy()
        id_list_valid['id'] = id_list_valid['id'].str[:-10] + 'validation'
        joblib.dump(id_list_eval, path_eval, compress=True)
        joblib.dump(id_list_valid, path_valid, compress=True)
    
    sub_list = []
    for id_list in [id_list_eval, id_list_valid]:
        sub = acc.merge(id_list)
        sub = pd.melt(
            sub,
            id_vars=INDEX_COLUMNS + ['aggregation_level'],
            var_name='d',
            value_name='sales'
        )
        sub['sales'] = sub['sales'].astype('float64')
        agg0 = sub[sub['aggregation_level'] == 0].copy()
        agg1 = sub[sub['aggregation_level'] == 1].copy()
        agg2 = sub[sub['aggregation_level'] == 2].copy()
        agg7 = sub[sub['aggregation_level'] == 7].copy()
        agg8 = sub[sub['aggregation_level'] == 8].copy()
        item = sub[sub['aggregation_level'] == 11].copy()
        def fit_sales(df1, df2, agg_column):
            df2 = df2.merge(df1[agg_column + ['d', 'sales']], on=agg_column + ['d'])
            df2['sum_sales'] = df2.groupby(agg_column + ['d'])['sales_x'].transform('sum')
            df2['sales'] = df2['sales_x'] / df2['sum_sales'] * df2['sales_y']
            return df2
        agg1 = fit_sales(agg0, agg1, AGGREGATION_LEVELS[0])
        agg2 = fit_sales(agg1, agg2, AGGREGATION_LEVELS[1])
        agg7 = fit_sales(agg2, agg7, AGGREGATION_LEVELS[2])
        agg8 = fit_sales(agg7, agg8, AGGREGATION_LEVELS[7])
        item = fit_sales(agg8, item, AGGREGATION_LEVELS[8])
        item = item.pivot(index='id', columns='d', values='sales')
        pred_columns = [f'F{i+1}' for i in range(28)]
        item = item[pred_columns].reset_index()
        sub_list.append(item)
    acc = pd.concat(sub_list)
    return acc

def main(run_name):
    args = parse_args(run_name)
    logging.info(f'agg file = {args.agg_datetime}, each file = {args.each_datetime}')
    save_dir = pathlib.Path('../data/submissions')
    agg_acc = joblib.load(save_dir / f'agg_submit-acc-{args.agg_datetime}.joblib')
    agg_unc = joblib.load(save_dir / f'agg_submit-unc-{args.agg_datetime}.joblib')
    each_acc = joblib.load(save_dir / f'each_submit-acc-{args.each_datetime}.joblib')
    each_unc = joblib.load(save_dir / f'each_submit-unc-{args.each_datetime}.joblib')
    sub_acc = pd.concat([agg_acc, each_acc])
    sub_unc = pd.concat([agg_unc, each_unc])
    sub_acc = modify_acc(sub_acc)
    sample_sub_acc = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
    sample_sub_unc = pd.read_csv('../input/m5-forecasting-uncertainty/sample_submission.csv')
    sub_acc = sample_sub_acc[['id']].merge(sub_acc)
    sub_unc = sample_sub_unc[['id']].merge(sub_unc)
    assert len(sample_sub_acc) == len(sub_acc)
    assert len(sample_sub_unc) == len(sub_unc)
    now = pd.Timestamp.now()
    sub_acc.to_csv(f'../submissions/acc-{now:%Y%m%d}-{now:%H%M%S}.csv', index=False, float_format='%.3g')
    sub_unc.to_csv(f'../submissions/unc-{now:%Y%m%d}-{now:%H%M%S}.csv', index=False, float_format='%.3g')


if __name__ == "__main__":
    run_name = init(__file__)
    try:
        main(run_name)
    except:
        logging.exception('exception')
    finally:
        logging.info('end')
