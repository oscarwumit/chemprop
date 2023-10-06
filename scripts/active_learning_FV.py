import os

import numpy as np
import pandas as pd
import pickle
from chemprop.args import TrainArgs, PredictArgs, get_checkpoint_paths
from chemprop.data import preprocess_smiles_columns
from chemprop.train import cross_validate, run_training, make_predictions


def select_data(df, criterion, size):
    if criterion == 'random':
        df_selected = df.sample(n=size, random_state=0)
    elif criterion == 'ensemble_variance':
        df_selected = df.sort_values(by='unc', ascending=False)
        df_selected = df_selected.head(n=size)
    else:
        return ValueError(f'criterion {criterion} not in defined list')
    return df_selected


if __name__ == '__main__':
    # set arguments for active learning manually below
    # on a longer term this could be added to the TAP args
    # training arguments are read from args
    active_learning_steps = 3
    data_selection_criterion = 'ensemble_variance'  # others are 'ensemble_variance'
    data_selection_fixed_amount = None  # number of training points you want to add, if None amount adjusts
    data_selection_variable_amount = 0.1  # fraction of the training size to be added

    path_experimental = '/home/fhvermei/Software/MIT2/chemprop/chemprop/data_FV/qm9_experimental.csv'  # path to data that can be added
    fixed_experimental_size = None  # experimental data set size, if None amount adjusts with data added
    variable_experimental_size_factor = 100.  # x times the size of the data to be added

    path_test = '/home/fhvermei/Software/MIT2/chemprop/chemprop/data_FV/qm9_test.csv'  # path to external test set

    train_args = TrainArgs().parse_args()
    path_training = train_args.data_path
    path_results = train_args.save_dir
    if not os.path.exists(path_results):
        os.mkdir(path_results)
    df_training = pd.read_csv(path_training)
    df_experimental_all = pd.read_csv(path_experimental)
    # double check here are no duplicates in experimental data. Assume there are no duplicates in training and test
    df_experimental_all = df_experimental_all.drop_duplicates(keep=False)
    df_test = pd.read_csv(path_test)
    df_experimental_all_results = pd.DataFrame()

    # set some args
    train_args.split_sizes = [0.9, 0.1, 0.0]  # we have a separate test set
    #train_args.ensemble_size = 2
    #train_args.num_folds = 2
    #train_args.epochs = 10

    for al_run in range(active_learning_steps):
        # make training and experimental dfs and select size of data to be added

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
        train_args.save_dir = os.path.join(path_results, f'models_run_{al_run}')
        cross_validate(args=train_args, train_func=run_training)

        # predict the experimental set
        predict_args = PredictArgs()
        predict_args.number_of_molecules = train_args.number_of_molecules
        predict_args.checkpoint_dir = train_args.save_dir
        predict_args.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=predict_args.checkpoint_path,
            checkpoint_paths=predict_args.checkpoint_paths,
            checkpoint_dir=predict_args.checkpoint_dir,
        )

        predict_args.uncertainty_method = 'ensemble'
        predict_args.test_path = os.path.join(path_results, f'temp_in.csv')
        predict_args.smiles_columns = preprocess_smiles_columns(
            path=predict_args.test_path,
            smiles_columns=predict_args.smiles_columns,
            number_of_molecules=predict_args.number_of_molecules,
        )
        predict_args.preds_path = os.path.join(path_results, f'temp_out.csv')
        preds, unc = make_predictions(args=predict_args, return_uncertainty=True)
        df_experimental[f'preds'] = np.ravel(preds)
        df_experimental[f'unc'] = np.ravel(unc)
        # df_experimental.to_csv(os.path.join(path_results, f'experimental_{al_run}.csv'), index=False)
        # now experimental sets are not saved, only the data selected from these sets

        # predict the test set
        predict_args.test_path = path_test
        predict_args.smiles_columns = preprocess_smiles_columns(
            path=predict_args.test_path,
            smiles_columns=predict_args.smiles_columns,
            number_of_molecules=predict_args.number_of_molecules,
        )
        predict_args.preds_path = os.path.join(path_results, f'temp_out.csv')
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
        train_args.data_path = path_training
        df_experimental_all = pd.concat([df_experimental_all, df_selected, df_selected], join='inner').drop_duplicates(keep=False)
        df_experimental_all_results = pd.concat([df_experimental_all_results, df_selected])
        print('run completed')
        if al_run % 10 == 0:
            df_experimental_all_results_temp = pd.concat([df_experimental_all_results, df_experimental_all])
            with open(os.path.join(path_results, f'experimental_results.pickle'), 'wb') as f:
                pickle.dump(df_experimental_all_results_temp, f)

df_experimental_all_results = pd.concat([df_experimental_all_results, df_experimental_all])
with open(os.path.join(path_results, f'experimental_results.pickle'), 'wb') as f:
    pickle.dump(df_experimental_all_results, f)
