import os
import gc
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
import torch
import hydra
from hydra.utils import get_original_cwd
import mlflow

from util import TARGET, seed_everything, reduce_mem_usage
from model import *


def dump(acc, unc, name):
    acc = reduce_mem_usage(acc)
    unc = reduce_mem_usage(unc)
    save_dir = pathlib.Path('../data/submissions')
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    now = pd.Timestamp.now()
    acc_name = f'{name}-acc-{now:%Y%m%d}-{now:%H%M%S}.joblib'
    unc_name = f'{name}-unc-{now:%Y%m%d}-{now:%H%M%S}.joblib'
    joblib.dump(acc, save_dir / acc_name, compress=True)
    joblib.dump(unc, save_dir / unc_name, compress=True)


def predict(args, module, criterion, data_dict, weight, te, evaluation=False):
    remove_last4w = 0 if evaluation else 1
    testset = M5TestDataset(data_dict, weight, te, remove_last4w=remove_last4w)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    device = next(iter(module.parameters())).device
    data_id_list = []
    y_pred_list = []
    samples_list = []
    for batch in test_loader:
        with torch.no_grad():
            module.eval()
            data_id, (x1, y1, x2, te), _, _ = batch
            data_id_list += data_id
            x1, y1, x2, te = x1.to(device), y1.to(device), x2.to(device), te.to(device)
            params = module((x1, y1, x2, te))
            y_pred = criterion.predict(params)
            y_pred_list.append(y_pred)
            samples = criterion.sample(params, 1000)
            samples_list.append(samples)
    y_pred = torch.cat(y_pred_list, dim=0).cpu().numpy()
    samples = torch.cat(samples_list, dim=1).cpu().numpy()
    quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
    pred_columns = [f'F{i+1}' for i in range(28)]
    if args.each:
        submission_accuracy = pd.DataFrame(y_pred, index=pd.Index(data_id_list, name='id'))
        submission_accuracy.columns = pred_columns
        submission_accuracy = submission_accuracy.reset_index()
        if evaluation:
            submission_accuracy['id'] = submission_accuracy['id'].str[:-10] + 'evaluation'
        else:
            submission_accuracy['id'] = submission_accuracy['id'].str[:-10] + 'validation'
        q = np.quantile(samples, quantiles, axis=0)
        q = q.transpose((1, 0, 2))  # quantile * id * date -> id * quantile * date
        q = q.reshape(-1, 28)
        multi_index = pd.MultiIndex.from_product([data_id_list, quantiles], names=['id', 'quantile'])
        df_q = pd.DataFrame(q, index=multi_index)
        df_q.columns = pred_columns
        df_q = df_q.reset_index()
        if evaluation:
            df_q['id'] = df_q['id'].str[:-10] + df_q['quantile'].map('{:.3f}'.format) + '_evaluation'
        else:
            df_q['id'] = df_q['id'].str[:-10] + df_q['quantile'].map('{:.3f}'.format) + '_validation'
        df_q.drop('quantile', axis=1, inplace=True)
        submission_uncertainty = df_q
    else:
        sales_transformers = joblib.load('../data/05_preprocess/agg_item/sales_transformers.joblib')
        pred_list = []
        q_list = []
        for i, data_id in enumerate(data_id_list):
            scaler = sales_transformers[data_id]
            inverse_pred = scaler.inverse_transform(y_pred[i, :].reshape(-1, 1)).reshape(1, 28)
            inverse = scaler.inverse_transform(samples[:, i, :].reshape(-1, 1)).reshape(1000, 28)
            df_pred = pd.DataFrame(inverse_pred)
            df_pred.columns = pred_columns
            if evaluation:
                df_pred.insert(0, 'id', data_id[:-10] + 'evaluation')
            else:
                df_pred.insert(0, 'id', data_id[:-10] + 'validation')
            pred_list.append(df_pred)
            q = np.quantile(inverse, quantiles, axis=0)
            df_q = pd.DataFrame(q, index=quantiles).reset_index()
            df_q.columns = ['quantile'] + pred_columns
            if evaluation:
                id_name = data_id[:-10] + df_q['quantile'].map('{:.3f}'.format) + '_evaluation'
            else:
                id_name = data_id[:-10] + df_q['quantile'].map('{:.3f}'.format) + '_validation'
            df_q.insert(0, 'id', id_name)
            df_q.drop('quantile', axis=1, inplace=True)
            q_list.append(df_q)
        submission_accuracy = pd.concat(pred_list).reset_index(drop=True)
        submission_uncertainty = pd.concat(q_list).reset_index(drop=True)
    return submission_accuracy, submission_uncertainty


def train(args):
    os.chdir(get_original_cwd())
    run_name = 'each' if args.each else 'agg'
    run_name += '_submit' if args.submit else '_cv'
    logging.info('start ' + run_name)
    seed_everything(args.seed)
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
    logging.info('data loaded')

    if args.submit:
        logging.info('train for submit')
        # train model for submission
        index = 1 if args.useval else 2
        valid_term = 2
        train_index = index if args.patience == 0 else (index+valid_term)
        trainset = M5Dataset(
            v_sales_dict, data_count, features, weight, te,
            remove_last4w=train_index, min_data_4w=0, over_sample=args.over_sample
        )
        validset = M5ValidationDataset(trainset.data_dict, weight, te, remove_last4w=index, term=valid_term)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            worker_init_fn=get_worker_init_fn(args.seed)
        )
        valid_loader = torch.utils.data.DataLoader(
            validset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        model = M5MLPLSTMModel(is_cats, dims, n_hidden=args.n_hidden, dropout=args.dropout, use_te=args.use_te)
        criterion = M5Distribution(dist=args.dist, df=args.df)
        module = M5LightningModule(model, criterion, train_loader, valid_loader, None, args)
        trainer = M5Trainer(args.experiment, run_name, args.max_epochs, args.min_epochs, args.patience, args.val_check)
        trainer.fit(module)
        trainer.logger.experiment.log_artifact(trainer.logger.run_id, trainer.checkpoint_callback.kth_best_model)

        logging.info('predict')
        module.load_state_dict(torch.load(trainer.checkpoint_callback.kth_best_model)['state_dict'])
        # for reproducibility
        dmp_filename = '../data/cuda_rng_state_each.dmp' if args.each else '../data/cuda_rng_state_agg.dmp'
        torch.save(torch.cuda.get_rng_state(), dmp_filename)
        trainer.logger.experiment.log_artifact(trainer.logger.run_id, dmp_filename)
        val_acc, val_unc = predict(args, module, criterion, trainset.data_dict, weight, te, evaluation=False)
        eva_acc, eva_unc = predict(args, module, criterion, trainset.data_dict, weight, te, evaluation=True)
        submission_accuracy = pd.concat([val_acc, eva_acc])
        submission_uncertainty = pd.concat([val_unc, eva_unc])
        dump(submission_accuracy, submission_uncertainty, run_name)

    else:
        # local CV
        folds = list(range(3, -1, -1))  # [3, 2, 1, 0]
        for fold in folds:
            logging.info(f'train FOLD [{4-fold}/{len(folds)}]')
            valid_term = 2
            if args.patience == 0:
                train_index = (fold + 1) * valid_term + 1
                valid_index = (fold + 1) * valid_term + 1
                test_index = fold * valid_term + 1
            else:
                train_index = (fold + 2) * valid_term + 1
                valid_index = (fold + 1) * valid_term + 1
                test_index = fold * valid_term + 1
            trainset = M5Dataset(v_sales_dict, data_count, features, weight, te, remove_last4w=train_index, over_sample=args.over_sample)
            validset = M5ValidationDataset(trainset.data_dict, weight, te, remove_last4w=valid_index, term=valid_term)
            testset = M5TestDataset(trainset.data_dict, weight, te, remove_last4w=test_index, term=valid_term)
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                worker_init_fn=get_worker_init_fn(args.seed)
            )
            valid_loader = torch.utils.data.DataLoader(
                validset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
            )
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
            )
            model = M5MLPLSTMModel(is_cats, dims, n_hidden=args.n_hidden, dropout=args.dropout, use_te=args.use_te)
            criterion = M5Distribution(dist=args.dist, df=args.df)
            module = M5LightningModule(model, criterion, train_loader, valid_loader, test_loader, args)
            fold_name = f'_{4-fold}-{len(folds)}'
            trainer = M5Trainer(args.experiment, run_name+fold_name, args.max_epochs, args.min_epochs, args.patience, args.val_check)
            trainer.fit(module)
            trainer.logger.experiment.log_artifact(trainer.logger.run_id, trainer.checkpoint_callback.kth_best_model)

            logging.info(f'test FOLD [{4-fold}/{len(folds)}]')
            module.load_state_dict(torch.load(trainer.checkpoint_callback.kth_best_model)['state_dict'])
            trainer.test()
            del trainset, validset, testset, train_loader, valid_loader, test_loader, model, criterion, module, trainer
            gc.collect()


@hydra.main(config_path='config.yaml')
def main(args):
    logging.info('start')
    try:
        train(args)
    except:
        logging.exception('exception')
    finally:
        logging.info('end')


if __name__ == "__main__":
    main()
