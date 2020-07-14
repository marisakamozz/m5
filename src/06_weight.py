import logging
import pathlib
import joblib

from util import init


def dump(df, name):
    save_dir = pathlib.Path('../data/06_weight')
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    joblib.dump(df, save_dir / f'{name}.joblib', compress=True)


def calc_weight(v_sales):
    v_sales['sort_key'] = v_sales['d'].str[2:].apply(lambda x: int(x))
    v_sales = v_sales[v_sales['sort_key'] >= 1886].copy()
    v_sales = v_sales[v_sales['sort_key'] <= 1913].copy()
    sum_sales = v_sales.groupby('id').agg({'aggregation_level': 'first', 'sales': 'sum'})
    sum_sales['total_sales'] = sum_sales.groupby('aggregation_level')['sales'].transform('sum')
    sum_sales = sum_sales.reset_index().sort_values(['aggregation_level', 'id']).reset_index(drop=True)
    sum_sales['weight'] = sum_sales['sales'] / sum_sales['total_sales']
    sum_sales['weight'] = sum_sales['weight'] / sum_sales['weight'].mean()
    return sum_sales.set_index('id')['weight'].to_dict()


def main(run_name):
    v_sales_agg = joblib.load('../data/04_agg/v_sales_agg.joblib')
    weight_agg = calc_weight(v_sales_agg)
    dump(weight_agg, 'weight_agg')

    v_sales_each = joblib.load('../data/04_agg/v_sales_each.joblib')
    weight_each = calc_weight(v_sales_each)
    dump(weight_each, 'weight_each')


if __name__ == "__main__":
    run_name = init(__file__)
    try:
        main(run_name)
    except:
        logging.exception('exception')
    finally:
        logging.info('end')
