import os
from random import Random

import numpy as np
import pandas as pd
import pickle

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from chemprop.args import TrainArgs, PredictArgs, get_checkpoint_paths, FingerprintArgs, ActiveLearningArgs
from chemprop.data import preprocess_smiles_columns
from chemprop.train import cross_validate, run_training, make_predictions
from chemprop.train.molecule_fingerprint import molecule_fingerprint


def split_calibration_set(args=None, trainingset=None):
    """function to reproduce data splitting from cross_validation"""
    """try to avoid making a chemprop data.MoleculeDataset object since that requires converting smiles"""
    if args.split_type != 'random':
        return ValueError('calibration only works with random split types')
    df_calibration = pd.DataFrame()
    init_seed = args.seed
    sizes = args.split_sizes
    data = trainingset
    for fold_num in range(args.num_folds):
        seed = init_seed + fold_num
        random = Random(seed)
        indices = list(range(len(data)))
        random.shuffle(indices)
        train_val_size = int((sizes[0] + sizes[1]) * len(data))
        indices_for_calibration = indices[train_val_size:]
        _df_calibration = data.loc[indices_for_calibration]
        _df_calibration['fold_idx'] = fold_num
        df_calibration = pd.concat([df_calibration, _df_calibration])
    return df_calibration


def process_predict_args(args, path_results, type, num_fold=None):
    if 'fp' in type:
        predict_args = FingerprintArgs()
    else:
        predict_args = PredictArgs()

    if type == 'test':
        test_path = args.path_test
    elif type =='exp' or type =='fp_exp' or type == 'cal':
        test_path = os.path.join(path_results, f'temp_in.csv')
    elif type == 'fp_train':
        test_path = args.data_path
    else:
        return ValueError('type unkown to parse prediction args')
    
    if 'fp' in type:
        preds_path = os.path.join(path_results, f'{type}.csv')
    else:
        preds_path = os.path.join(path_results, f'temp_out.csv')

    if 'cal' in type:
        checkpoint_dir = os.path.join(args.save_dir, f'fold_{num_fold}')
    else:
        checkpoint_dir = args.save_dir
    if args.data_selection_criterion in ['mve_var', 'mve_var_scaled']:
        uncertainty_method = 'mve'
    elif args.data_selection_criterion in ['evi_var', 'evi_var_scaled']:
        uncertainty_method = 'evidential_epistemic'
    elif args.data_selection_criterion in ['random', 'on_the_fly_clustering', 'on_the_fly_clustering_silhouette', 'on_the_fly_clustering_in_cluster_dist_ratio', 'latent_dist']:
        uncertainty_method = None
    else:
        uncertainty_method = 'ensemble'

    args = [
        "--number_of_molecules", str(args.number_of_molecules),
        "--checkpoint_dir", checkpoint_dir,
        "--test_path", test_path,
        "--preds_path", preds_path,
        "--num_workers", "0", # experiencing issues with multiprocessing for predictions...
    ]

    if uncertainty_method is not None:
        args.extend([
            "--uncertainty_method", uncertainty_method,
        ])

    predict_args.parse_args(args)
    return predict_args


def select_data(df, criterion, size):
    if criterion == 'random':
        df_selected = df.sample(n=size, random_state=0)
        return df_selected
    elif criterion in ['ens_var', 'mve_var', 'evi_var']:
        df_selected = df.sort_values(by='unc', ascending=False)
        df_selected = df_selected.head(n=size)
        return df_selected
    elif criterion in ['ens_var_scaled', 'mve_var_scaled', 'evi_var_scaled']:
        df['scaled_variance'] = df[f'unc'] / (np.abs(df[f'preds']) + np.mean(np.abs(df[f'preds'])))
        df_selected = df.sort_values(by=f'scaled_variance', ascending=False)
        df_selected = df_selected.head(n=size)
        return df_selected
    elif criterion == 'latent_dist':
        df_selected = df.sort_values(by='avg_min_dist', ascending=False)
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
    elif criterion == 'on_the_fly_clustering_silhouette':
        df_selected = df.sort_values(by='avg_a_b_ratio', ascending=False)
        df_selected = df_selected.head(n=size)
        return df_selected
    elif criterion == 'on_the_fly_clustering_in_cluster_dist_ratio':
        df_selected = df.sort_values(by='avg_in_cluster_dist_ratio', ascending=False)
        df_selected = df_selected.head(n=size)
        return df_selected
    elif criterion == 'on_the_fly_clustering_weight':
        df = df.sort_values(by='unc', ascending=False)
        sum_factors = df.unc.sum()
        df_selected = pd.DataFrame()
        ensemble_size = max([int(column.split("cluster_")[1]) for column in df.columns if 'cluster_' in column]) + 1
        size_per_model = int(size/ensemble_size)
        for model_idx in range(ensemble_size):
            clusters = df[f'cluster_{model_idx}'].unique()
            for c in clusters:
                df_temp = df[df[f'cluster_{model_idx}'] == c]
                c_factor = np.sum(df_temp.unc.values)
                size_to_add = round(c_factor/sum_factors*size_per_model)
                df_temp = df_temp.head(n=size_to_add)
                df_selected = pd.concat([df_selected, df_temp])
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
    if "on_the_fly_clustering" in data_selection_criterion:
        if args.use_pca_for_clustering:
            if args.pca_number_of_components is None and args.pca_fraction_of_variance_explained is None:
                raise ValueError('You need to specify either the number of components or the fraction of variance explained '
                                 'for the PCA.')
            elif args.pca_number_of_components is not None and args.pca_fraction_of_variance_explained is not None:
                raise ValueError('You need to specify either the number of components or the fraction of variance explained '
                                 'for the PCA, not both.')
            elif args.pca_number_of_components is not None:
                n_components = args.pca_number_of_components
            elif args.pca_fraction_of_variance_explained is not None:
                if not (0 < args.pca_fraction_of_variance_explained <= 1):
                    raise ValueError('The fraction of variance explained for the PCA needs to be between 0 and 1.')
                n_components = args.pca_fraction_of_variance_explained

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

    #todo, this can easily cause errors when other arguments are added. Be careful! And maybe change this to a better way
    if 'cluster' in data_selection_criterion and not 'clustering' in data_selection_criterion:
        if not 'cluster' in df_experimental_all.columns:
            raise ValueError(f'The experimental dataset needs a column called cluster if you '
                             f'use the {data_selection_criterion} method.')

    if "on_the_fly_clustering" in data_selection_criterion:
        clustering_results = dict()

    for al_run in range(active_learning_steps):
        print(f'Active learning run {al_run} of {active_learning_steps}...')
        
        size_training = len(df_training.index)
        size_data_selection = int(data_selection_fixed_amount) if data_selection_fixed_amount \
            else round(size_training * data_selection_variable_amount)
            
        if not fixed_experimental_size:
            size_experimental = int(variable_experimental_size_factor * size_data_selection)
        else:
            size_experimental = int(fixed_experimental_size)

        df_experimental = df_experimental_all.sample(n=size_experimental, random_state=al_run)

        # split calibration set, needs to be done before training or seed will be adjusted
        if 'calibration' in data_selection_criterion:
            df_calibration = split_calibration_set(args=args, trainingset=df_training)

        # train the model
        print('Training model...')
        args.save_dir = os.path.join(path_results, f'models_run_{al_run}')
        cross_validate(args=args, train_func=run_training)
        #reset save_dir because updated during training
        args.save_dir = os.path.join(path_results, f'models_run_{al_run}')

        df_calibration_results = pd.DataFrame()
        # predict the calibration set
        if 'calibration' in data_selection_criterion:
            print('Predicting calibration sets...')
            for fold_num in range(args.num_folds):
                _df_calibration = df_calibration[df_calibration.fold_idx == fold_num]
                _df_calibration.to_csv(os.path.join(path_results, f'temp_in.csv'), index=False)
                predict_args = process_predict_args(args, path_results, type='cal', num_fold=fold_num)
                preds, unc = make_predictions(args=predict_args, return_uncertainty=True)
                _df_calibration[f'preds'] = np.ravel(preds)
                _df_calibration[f'unc'] = np.ravel(unc)
                df_calibration_results = pd.concat([df_calibration_results, _df_calibration])
                #todo do we need to save these results?

        # predict the experimental set
        print('Predicting experimental set...')

        # save the experimental file because chemprop will only read a csv file
        df_experimental.to_csv(os.path.join(path_results, f'temp_in.csv'), index=False)
        predict_args = process_predict_args(args, path_results, type='exp')
        print(predict_args)
        preds, unc = make_predictions(args=predict_args, return_uncertainty=True)
        df_experimental[f'preds'] = np.ravel(preds)
        df_experimental[f'unc'] = np.ravel(unc)
        if 'on_the_fly_clustering' in data_selection_criterion or 'latent_dist' in data_selection_criterion:
            # make the fingerprints for the exp data
            fingerprint_args = process_predict_args(args, path_results, type='fp_exp')
            molecule_fingerprint(args=fingerprint_args)
            df_fp_exp = pd.read_csv(os.path.join(path_results, f'fp_exp.csv'))
            # make the fingerprints for the training data
            fingerprint_args = process_predict_args(args, path_results, type='fp_train')
            molecule_fingerprint(args=fingerprint_args)
            df_fp = pd.read_csv(os.path.join(path_results, f'fp_train.csv'))

        if 'on_the_fly_clustering' in data_selection_criterion:

            for model_idx in range(args.ensemble_size):

                if args.ensemble_size > 1:
                    columns = [c for c in df_fp.columns if f'mol_{args.fingerprint_idx}_model_{model_idx}' in c]
                else:
                    columns = [c for c in df_fp.columns if f'mol_{args.fingerprint_idx}' in c]

                df_temp = pd.DataFrame(df_fp, columns=columns)
                if args.use_pca_for_clustering:
                    pca = PCA(n_components=n_components)
                    components = pca.fit_transform(df_temp)
                    for i in range(components.shape[1]):
                        df_fp[f'pc_{i + 1}_{model_idx}'] = components[:, i]
                    for_clustering = components
                else:
                    for_clustering = df_temp
                kmeanModel = KMeans(n_clusters=args.number_of_clusters, random_state=0, n_init='auto').fit(for_clustering)
                df_fp[f'cluster_{model_idx}'] = kmeanModel.labels_
                max_in_cluster_dists = np.array([np.min(cdist(for_clustering[kmeanModel.labels_==i], kmeanModel.cluster_centers_[[i]])) for i in range(args.number_of_clusters)])

                # apply clustering
                if args.ensemble_size > 1:
                    columns = [c for c in df_fp_exp.columns if f'mol_{args.fingerprint_idx}_model_{model_idx}' in c]
                else:
                    columns = [c for c in df_fp_exp.columns if f'mol_{args.fingerprint_idx}' in c]

                df_temp_exp = pd.DataFrame(df_fp_exp, columns=columns)
                if args.use_pca_for_clustering:
                    components = pca.transform(df_temp_exp)
                    for i in range(components.shape[1]):
                        df_fp_exp[f'pc_{i + 1}_{model_idx}'] = components[:, i]
                    for_clustering = components
                else:
                    for_clustering = df_temp_exp
                clustering = kmeanModel.predict(for_clustering)
                df_fp_exp[f'cluster_{model_idx}'] = clustering
                if data_selection_criterion == 'on_the_fly_clustering':
                    df_fp_exp[f'min_distance_{model_idx}'] = np.min(
                        cdist(for_clustering, kmeanModel.cluster_centers_, 'euclidean'), axis=1)
                    df_experimental[f'min_distance_{model_idx}'] = df_fp_exp[f'min_distance_{model_idx}'].values
                    df_experimental[f'norm_min_distance_{model_idx}'] = (df_experimental[f'min_distance_{model_idx}']/
                                                                df_experimental[f'min_distance_{model_idx}'].max())
                elif data_selection_criterion == 'on_the_fly_clustering_silhouette':
                    cdists = cdist(for_clustering, kmeanModel.cluster_centers_, 'euclidean')
                    cdists = np.sort(cdists, axis=1)
                    a_b_ratios = cdists[:, 0]/cdists[:, 1]
                    df_experimental[f'min_distance_{model_idx}'] = cdists[:, 0]
                    df_experimental[f'second_min_distance_{model_idx}'] = cdists[:, 1]
                    df_experimental[f'a_b_ratio_{model_idx}'] = a_b_ratios
                elif data_selection_criterion == 'on_the_fly_clustering_in_cluster_dist_ratio':
                    max_in_cluster_dists = np.array([max_in_cluster_dists[cluster] for cluster in clustering])
                    min_dists = np.min(
                        cdist(for_clustering, kmeanModel.cluster_centers_, 'euclidean'), axis=1)
                    df_experimental[f'max_in_cluster_dist_{model_idx}'] = max_in_cluster_dists
                    df_experimental[f'min_distance_{model_idx}'] = min_dists
                    df_experimental[f'in_cluster_dist_ratio_{model_idx}'] = min_dists/max_in_cluster_dists
                elif data_selection_criterion == 'on_the_fly_clustering_weight':
                    df_experimental[f'cluster_{model_idx}'] = clustering

            if data_selection_criterion == 'on_the_fly_clustering':
                columns = [f'norm_min_distance_{model_idx}' for model_idx in range(args.ensemble_size)]
                df_experimental[f'avg_norm_min_distance'] = df_experimental[columns].sum(axis=1)
            elif data_selection_criterion == 'on_the_fly_clustering_silhouette':
                columns = [f'a_b_ratio_{model_idx}' for model_idx in range(args.ensemble_size)]
                df_experimental[f'avg_a_b_ratio'] = df_experimental[columns].sum(axis=1)
            elif data_selection_criterion == 'on_the_fly_clustering_in_cluster_dist_ratio':
                columns = [f'in_cluster_dist_ratio_{model_idx}' for model_idx in range(args.ensemble_size)]
                df_experimental[f'avg_in_cluster_dist_ratio'] = df_experimental[columns].sum(axis=1)

        elif 'latent_dist' in data_selection_criterion:

            for model_idx in range(args.ensemble_size):

                if args.ensemble_size > 1:
                    columns = [c for c in df_fp.columns if f'mol_{args.fingerprint_idx}_model_{model_idx}' in c]
                else:
                    columns = [c for c in df_fp.columns if f'mol_{args.fingerprint_idx}' in c]

                df_temp = pd.DataFrame(df_fp, columns=columns)
                
                if args.ensemble_size > 1:
                    columns = [c for c in df_fp_exp.columns if f'mol_{args.fingerprint_idx}_model_{model_idx}' in c]
                else:
                    columns = [c for c in df_fp_exp.columns if f'mol_{args.fingerprint_idx}' in c]

                df_temp_exp = pd.DataFrame(df_fp_exp, columns=columns)

                # Fit NearestNeighbors model
                nbrs = NearestNeighbors(n_neighbors=args.number_of_knn).fit(df_temp)

                # Find indices and distances of nearest neighbors
                distances, _ = nbrs.kneighbors(df_temp_exp)

                # Calculate mean distances
                knn_min_dists = np.mean(distances, axis=1)
                
                df_experimental[f'min_dist_{model_idx}'] = knn_min_dists

            columns = [f'min_dist_{model_idx}' for model_idx in range(args.ensemble_size)]
            df_experimental[f'avg_min_dist'] = df_experimental[columns].sum(axis=1)
            

        # predict the test set
        print('Predicting test set...')
        if args.path_test is not None:
            predict_args = process_predict_args(args, path_results, type='test')
            print(predict_args)
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
        if (al_run+1) % (args.active_learning_steps/5) == 0:
            df_experimental_all_results_temp = pd.concat([df_experimental_all_results, df_experimental_all])
            with open(os.path.join(path_results, f'experimental_results.pickle'), 'wb') as f:
                pickle.dump(df_experimental_all_results_temp, f)
            if "on_the_fly_clustering" in data_selection_criterion:
                clustering_results[al_run] = (df_fp, df_fp_exp)
                with open(os.path.join(path_results, f'clustering_results.pickle'), 'wb') as f:
                    pickle.dump(clustering_results, f)


def active_learning() -> None:
    run_active_learning(args=ActiveLearningArgs().parse_args())


if __name__ == '__main__':
    active_learning()
