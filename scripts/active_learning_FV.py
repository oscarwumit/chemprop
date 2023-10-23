import os

import numpy as np
import pandas as pd
import pickle

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from chemprop.args import TrainArgs, PredictArgs, get_checkpoint_paths, FingerprintArgs, ActiveLearningArgs
from chemprop.data import preprocess_smiles_columns
from chemprop.train import cross_validate, run_training, make_predictions
from chemprop.train.molecule_fingerprint import molecule_fingerprint


def process_predict_args(args, path_results, type):
    if 'fp' in type:
        predict_args = FingerprintArgs()
    else:
        predict_args = PredictArgs()
    predict_args.number_of_molecules = args.number_of_molecules
    predict_args.checkpoint_dir = args.save_dir
    predict_args.checkpoint_paths = get_checkpoint_paths(
        checkpoint_path=predict_args.checkpoint_path,
        checkpoint_paths=predict_args.checkpoint_paths,
        checkpoint_dir=predict_args.checkpoint_dir,
    )

    predict_args.uncertainty_method = 'ensemble'
    if type == 'test':
        predict_args.test_path = args.path_test
    elif type =='exp' or type =='fp_exp':
        predict_args.test_path = os.path.join(path_results, f'temp_in.csv')
    elif type == 'fp_train':
        predict_args.test_path = args.data_path
    else:
        return ValueError('type unkown to parse prediction args')
    predict_args.smiles_columns = preprocess_smiles_columns(
        path=predict_args.test_path,
        smiles_columns=predict_args.smiles_columns,
        number_of_molecules=predict_args.number_of_molecules,
    )
    if 'fp' in type:
        predict_args.preds_path = os.path.join(path_results, f'{type}.csv')
    else:
        predict_args.preds_path = os.path.join(path_results, f'temp_out.csv')
    return predict_args


def select_data(df, criterion, size):
    if criterion == 'random':
        df_selected = df.sample(n=size, random_state=0)
        return df_selected
    elif criterion == 'ens_var':
        df_selected = df.sort_values(by='unc', ascending=False)
        df_selected = df_selected.head(n=size)
        return df_selected
    elif criterion == 'cluster_equal':
        if not 'cluster' in df.columns:
            raise ValueError('The experimental data needs a column called cluster')
        clusters = df['cluster'].unique()
        residual = size%len(clusters)
        df_selected = pd.DataFrame()
        for c in clusters:
            df_temp = df[df.cluster == c]
            df_temp = df_temp.sort_values(by='unc', ascending=False)
            number = int(size/len(clusters))
            number = number + 1 if residual>0 else number
            residual -= 1
            df_temp = df_temp.head(n=number)
            df_selected = pd.concat([df_selected, df_temp])
        return df_selected
    elif criterion == 'cluster_weight':
        if not 'cluster' in df.columns:
            raise ValueError('The experimental data needs a column called cluster')
        clusters = df['cluster'].unique()
        df = df.sort_values(by='unc', ascending=False)
        df = df.groupby('cluster').head(n=size).reset_index(drop=True)
        sum_factors = df.unc.sum()
        df_selected = pd.DataFrame()
        for c in clusters:
            df_temp = df[df.cluster == c]
            c_factor = np.sum(df_temp.unc.values)
            size_to_add = round(c_factor/sum_factors*size)
            df_temp = df_temp.head(n=size_to_add)
            df_selected = pd.concat([df_selected, df_temp])
        return df_selected
    elif criterion == 'on_the_fly_clustering':
        df_selected = df.sort_values(by='avg_norm_min_distance', ascending=False)
        df_selected = df_selected.head(n=size)
        return df_selected
    else:
        return ValueError(f'criterion {criterion} not in defined list')


def run_active_learning(args: ActiveLearningArgs):

    active_learning_steps = args.active_learning_steps
    data_selection_criterion = args.data_selection_criterion
    data_selection_fixed_amount = args.data_selection_fixed_amount
    data_selection_variable_amount = args.data_selection_variable_amount
    fixed_experimental_size = args.fixed_experimental_size
    variable_experimental_size_factor = args.variable_experimental_size_factor

    path_results = args.save_dir
    if not os.path.exists(path_results):
        os.mkdir(path_results)
    df_training = pd.read_csv(args.data_path)
    df_experimental_all = pd.read_csv(args.path_experimental)
    # double check here are no duplicates in experimental data. Assume there are no duplicates in training and test
    df_experimental_all = df_experimental_all.drop_duplicates(keep=False)
    if args.path_test is not None:
        df_test = pd.read_csv(args.path_test)

    df_experimental_all_results = pd.DataFrame()

    if 'cluster' in data_selection_criterion and not 'clustering' in data_selection_criterion:
        if not 'cluster' in df_experimental_all.columns:
            raise ValueError(f'The experimental dataset needs a column called cluster if you '
                             f'use the {data_selection_criterion} method.')
    if data_selection_criterion == 'on_the_fly_clustering':
        clustering_results = dict()

    for al_run in range(active_learning_steps):
        size_training = len(df_training.index)
        size_data_selection = int(data_selection_fixed_amount) if data_selection_fixed_amount \
            else round(size_training * data_selection_variable_amount)
            
        if not fixed_experimental_size:
            size_experimental = int(variable_experimental_size_factor * size_data_selection)
        else:
            size_experimental = int(fixed_experimental_size)

        df_experimental = df_experimental_all.sample(n=size_experimental, random_state=al_run)
        # safe the experimental file because chemprop will only read a csv file
        df_experimental.to_csv(os.path.join(path_results, f'temp_in.csv'), index=False)

        # train the model
        args.save_dir = os.path.join(path_results, f'models_run_{al_run}')
        cross_validate(args=args, train_func=run_training)

        # predict the experimental set
        predict_args = process_predict_args(args, path_results, type='exp')
        preds, unc = make_predictions(args=predict_args, return_uncertainty=True)
        df_experimental[f'preds'] = np.ravel(preds)
        df_experimental[f'unc'] = np.ravel(unc)
        if data_selection_criterion == 'on_the_fly_clustering':
            # make the fingerprints for the exp data
            fingerprint_args = process_predict_args(args, path_results, type='fp_exp')
            molecule_fingerprint(args=fingerprint_args)
            df_fp_exp = pd.read_csv(os.path.join(path_results, f'fp_exp.csv'))
            # make the fingerprints for the training data
            fingerprint_args = process_predict_args(args, path_results, type='fp_train')
            molecule_fingerprint(args=fingerprint_args)
            df_fp = pd.read_csv(os.path.join(path_results, f'fp_train.csv'))

            for model_idx in range(args.ensemble_size):
                df_temp = pd.DataFrame(df_fp, columns=[c for c in df_fp.columns if
                                                       f'mol_{args.fingerprint_idx}_model_{model_idx}' in c])
                if args.use_pca_for_clustering:
                    pca = PCA(n_components=args.pca_number_of_components)
                    components = pca.fit_transform(df_temp)
                    for i in range(args.pca_number_of_components):
                        df_fp[f'pc_{i + 1}_{model_idx}'] = components[:, i]
                    for_clustering = components
                else:
                    for_clustering = df_temp
                kmeanModel = KMeans(n_clusters=args.number_of_clusters, random_state=0, n_init='auto').fit(for_clustering)
                df_fp[f'cluster_{model_idx}'] = kmeanModel.labels_

                # apply clustering
                df_temp_exp = pd.DataFrame(df_fp_exp, columns=[c for c in df_fp_exp.columns if
                                                               f'mol_{args.fingerprint_idx}_model_{model_idx}' in c])
                if args.use_pca_for_clustering:
                    components = pca.transform(df_temp_exp)
                    for i in range(args.pca_number_of_components):
                        df_fp_exp[f'pc_{i + 1}_{model_idx}'] = components[:, i]
                    for_clustering = components
                else:
                    for_clustering = df_temp
                clustering = kmeanModel.predict(for_clustering)
                df_fp_exp[f'cluster_{model_idx}'] = clustering
                df_fp_exp[f'min_distance_{model_idx}'] = np.min(
                    cdist(for_clustering, kmeanModel.cluster_centers_, 'euclidean'), axis=1)
                df_experimental[f'min_distance_{model_idx}'] = df_fp_exp[f'min_distance_{model_idx}'].values
                df_experimental[f'norm_min_distance_{model_idx}'] = (df_experimental[f'min_distance_{model_idx}']/
                                                               df_experimental[f'min_distance_{model_idx}'].max())
            columns = [f'norm_min_distance_{model_idx}' for model_idx in range(args.ensemble_size)]
            df_experimental[f'avg_norm_min_distance'] = df_experimental[columns].sum(axis=1)
        # df_experimental.to_csv(os.path.join(path_results, f'experimental_{al_run}.csv'), index=False)
        # now experimental sets are not saved, only the data selected from these sets

        # predict the test set
        if args.path_test is not None:
            predict_args = process_predict_args(args, path_results, type='test')
            preds, unc = make_predictions(args=predict_args, return_uncertainty=True)
            df_test[f'preds_run{al_run}'] = np.ravel(preds)
            df_test[f'unc_run{al_run}'] = np.ravel(unc)
            with open(os.path.join(path_results, f'test_results.pickle'), 'wb') as f:
                pickle.dump(df_test, f)

        df_selected = select_data(df_experimental, data_selection_criterion, size_data_selection)
        df_selected[f'run'] = al_run
        df_training = pd.concat([df_training, df_selected], join='inner')
        df_training.to_csv(os.path.join(path_results, f'training_temp.csv'), index=False)
        path_training = os.path.join(path_results, f'training_temp.csv')
        df_training = pd.read_csv(path_training)
        args.data_path = path_training
        df_experimental_all = pd.concat([df_experimental_all, df_selected, df_selected], join='inner').drop_duplicates(keep=False)
        df_experimental_all_results = pd.concat([df_experimental_all_results, df_selected])
        print('run completed')
        if al_run % 2 == 0:
            df_experimental_all_results_temp = pd.concat([df_experimental_all_results, df_experimental_all])
            with open(os.path.join(path_results, f'experimental_results.pickle'), 'wb') as f:
                pickle.dump(df_experimental_all_results_temp, f)
            if data_selection_criterion == 'on_the_fly_clustering':
                clustering_results[al_run] = (df_fp, df_fp_exp)
                with open(os.path.join(path_results, f'clustering_results.pickle'), 'wb') as f:
                    pickle.dump(clustering_results, f)

    df_experimental_all_results = pd.concat([df_experimental_all_results, df_experimental_all])
    with open(os.path.join(path_results, f'experimental_results.pickle'), 'wb') as f:
        pickle.dump(df_experimental_all_results, f)

    if data_selection_criterion == 'on_the_fly_clustering':
        clustering_results[al_run] = (df_fp, df_fp_exp)
        with open(os.path.join(path_results, f'clustering_results.pickle'), 'wb') as f:
            pickle.dump(clustering_results, f)


def active_learning() -> None:
    run_active_learning(args=ActiveLearningArgs().parse_args())


if __name__ == '__main__':
    active_learning()
