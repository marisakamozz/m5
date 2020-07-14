import logging
import yaml
import joblib
import torch
import pandas as pd
import torch

from util import init, reduce_mem_usage
from model import *
from train import predict
from makesub import modify_acc


class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def fields(self):
        return self.obj

    def keys(self):
        return self.obj.keys()
    
    def items(self):
        return self.obj.items()


def predict_by_saved_model(config):
    args = AttributeDict(config)
    if args.each:
        v_sales_dict = joblib.load('../data/05_preprocess/each_item/v_sales_dict.joblib')
        data_count = joblib.load('../data/05_preprocess/each_item/data_count.joblib')
        dims = joblib.load('../data/05_preprocess/each_item/dims.joblib')
        weight = joblib.load('../data/06_weight/weight_each.joblib')
        te = joblib.load('../data/07_te/each_te.joblib')
    else:
        v_sales_dict = joblib.load('../data/05_preprocess/agg_item/v_sales_dict.joblib')
        data_count = joblib.load('../data/05_preprocess/agg_item/data_count.joblib')
        dims = joblib.load('../data/05_preprocess/agg_item/dims.joblib')
        weight = joblib.load('../data/06_weight/weight_agg.joblib')
        te = joblib.load('../data/07_te/agg_te.joblib')
    v_sales = next(iter(v_sales_dict.values()))
    drop_columns = ['sort_key', 'id', 'cat_id', 'd', 'release_date', 'date', 'weekday', 'year', 'week_of_month', 'holidy']
    if not args.use_prices:
        drop_columns += [
            'release_ago', 'sell_price', 'diff_price',
            'price_max', 'price_min', 'price_std', 'price_mean', 'price_trend', 'price_norm', 'diff_price_norm', 'price_nunique',
            'dept_max', 'dept_min', 'dept_std', 'dept_mean', 'price_in_dept', 'mean_in_dept',
            'cat_max', 'cat_min', 'cat_std', 'cat_mean', 'price_in_cat', 'mean_in_cat',
            'price_in_month', 'price_in_year',
        ]
    cat_columns = ['aggregation_level', 'item_id', 'dept_id', 'store_id', 'state_id', 'month', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'day_of_week']
    features = [col for col in v_sales.columns if col not in drop_columns + [TARGET]]
    is_cats = [col in cat_columns for col in features]
    cat_dims = []
    emb_dims = []
    for col in features:
        if col in cat_columns:
            cat_dims.append(dims['cat_dims'][col])
            emb_dims.append(dims['emb_dims'][col])
    dims = pd.DataFrame({
        'cat_dims': cat_dims,
        'emb_dims': emb_dims
    })
    train_index = 1 if args.useval else 2
    trainset = M5Dataset(
        v_sales_dict, data_count, features, weight, te,
        remove_last4w=train_index, min_data_4w=0, over_sample=args.over_sample
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        worker_init_fn=get_worker_init_fn(args.seed)
    )
    model = M5MLPLSTMModel(is_cats, dims, n_hidden=args.n_hidden, dropout=args.dropout, use_te=args.use_te)
    criterion = M5Distribution(dist=args.dist, df=args.df)
    module = M5LightningModule(model, criterion, train_loader, None, None, args)
    if torch.cuda.is_available():
        module = module.cuda()
    filename = '../models/each.ckpt' if args.each else '../models/agg.ckpt'
    module.load_state_dict(torch.load(filename)['state_dict'])
    device = next(iter(module.parameters())).device
    if args.each:
        cuda_rng_state = torch.load('../models/cuda_rng_state_each.dmp')
    else:
        cuda_rng_state = torch.load('../models/cuda_rng_state_agg.dmp')
    torch.cuda.set_rng_state(cuda_rng_state, device=device)
    val_acc, val_unc = predict(args, module, criterion, trainset.data_dict, weight, te, evaluation=False)
    eva_acc, eva_unc = predict(args, module, criterion, trainset.data_dict, weight, te, evaluation=True)
    pred_acc = reduce_mem_usage(pd.concat([val_acc, eva_acc]))
    pred_unc = reduce_mem_usage(pd.concat([val_unc, eva_unc]))
    return pred_acc, pred_unc


def main(run_name):
    with open('config.yaml') as file:
        config = yaml.safe_load(file)
    with open('trainer/agg_submit.yaml') as file:
        agg_config = yaml.safe_load(file)
    with open('trainer/each_submit.yaml') as file:
        each_config = yaml.safe_load(file)
    
    agg_config.update(config)
    each_config.update(config)

    logging.info('predict agg')
    agg_acc, agg_unc = predict_by_saved_model(agg_config)
    logging.info('predict each')
    each_acc, each_unc = predict_by_saved_model(each_config)

    logging.info('make submission')
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
