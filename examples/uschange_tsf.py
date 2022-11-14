"""
======================
Time Series Forecasting
======================

The following example shows how to fit a sample forecasting model
with AutoPyTorch. This is only a dummmy example because of the limited size of the dataset.
Thus, it could be possible that the AutoPyTorch model does not perform as well as a dummy predictor
"""
import os
import tempfile as tmp
import pathlib
import warnings
import copy
import numpy as np
import sys
from easydict import EasyDict
import torch
from torchviz import make_dot

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sktime.datasets import load_uschange
targets, features = load_uschange()
# start_times = [targets.index.to_timestamp()[0]]  # not used in uschange dataset
freq = '1Q'

exp_index='test_save_model'
exp_config = EasyDict(
    temporary_directory='./uschange/uschange_tmp_{}'.format(exp_index),
    output_directory='./uschange/uschange_out_{}'.format(exp_index),
    forecasting_horizon=3,
    func_eval_time_limit_secs=60 * 5,
    total_walltime_limit=60 * 60,
)

# Dataset optimized by APT-TS can be a list of np.ndarray / pd.DataFrame where each series represents an element in the
# list, or a single pd.DataFrame that records the series
# index information: to which series the timestep belongs? This id can be stored as the DataFrame's index or a separate
# column
# Within each series, we take the last forecasting_horizon as test targets. The items before that as training targets
# Normally the value to be forecasted should follow the training sets
y_train = [targets[: -exp_config.forecasting_horizon]]
y_test = [targets[-exp_config.forecasting_horizon:]]

# same for features. For uni-variant models, X_train, X_test can be omitted and set as None
X_train = [features[: -exp_config.forecasting_horizon]]
# Here x_test indicates the 'known future features': they are the features known previously, features that are unknown
# could be replaced with NAN or zeros (which will not be used by our networks). If no feature is known beforehand,
# we could also omit X_test
known_future_features = list(features.columns)
X_test = [features[-exp_config.forecasting_horizon:]]

# print(y_train[0].shape, y_test[0].shape, X_train[0].shape, X_test[0].shape)
# (184,) (3,) (184, 4) (3, 4)

print("data ok")

from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
############################################################################
# Build and fit a forecaster
# ==========================
api = TimeSeriesForecastingTask(
    temporary_directory=exp_config.temporary_directory,
    output_directory=exp_config.output_directory,
    delete_tmp_folder_after_terminate=False,
    delete_output_folder_after_terminate=False,
)
# api.set_pipeline_options(device="cuda") 
print(api.get_pipeline_options())


############################################################################
# Search for an ensemble of machine learning algorithms
# =====================================================
api.search(
    X_train=X_train,
    y_train=copy.deepcopy(y_train),
    X_test=X_test,
    y_test=copy.deepcopy(y_test),
    optimize_metric='mean_MASE_forecasting',
    n_prediction_steps=exp_config.forecasting_horizon,
    memory_limit=16 * 1024,  # Currently, forecasting models use much more memories
    freq=freq,
    # start_times=start_times,
    func_eval_time_limit_secs=exp_config.func_eval_time_limit_secs,
    total_walltime_limit=exp_config.total_walltime_limit,
    min_num_test_instances=1000,  # proxy validation sets. This only works for the tasks with more than 1000 series
    known_future_features=known_future_features,
)

print('search ok')

from autoPyTorch.datasets.time_series_dataset import TimeSeriesSequence

test_sets = []
# We could construct test sets from scratch
# for feature, future_feature, target, start_time in zip(X_train, X_test,y_train, start_times):
for feature, future_feature, target in zip(X_train, X_test, y_train):
    test_sets.append(
        TimeSeriesSequence(X=feature.values,
                           Y=target.values,
                           X_test=future_feature.values,
                        #    start_time=start_time,
                           is_test_set=True,
                           # additional information required to construct a new time series sequence
                           **api.dataset.sequences_builder_kwargs
                           )
    )
# Alternatively, if we only want to forecast the value after the X_train, we could directly ask datamanager to
# generate a test set:
# test_sets2 = api.dataset.generate_test_seqs()

################################################################################
# Print useful information
# (use try-except in case some are not functional for Time Series task)
# ======================================================================
try:
    y_pred = api.predict(test_sets)
    np.savetxt(exp_config.output_directory + "/y_pred.txt", y_pred, delimiter=',')
    print('== y_pred saved in {}'.format(exp_config.output_directory + "/y_pred.txt"))
except Exception as e:
    print('====== api.predict() fails ======')
try:
    print('== models:', api.show_models())
except Exception as e:
    print('show_models: {}'.format(e))
try:
    print('== stat  :', api.sprint_statistics())
except Exception as e:
    print('sprint_statistics: {}'.format(e))
try:
    print('== socre :', api.score(y_pred, y_test))  # , sp=freq, n_prediction_steps=exp_config.forecasting_horizon))
except Exception as e:
    print('score: {}'.format(e))
try:
    print('== models:', api.model)
except Exception as e:
    print('get_pytorch_model: {}'.format(e))

############################################################################
# Visualize best models with torchviz in `tmp_path/best_models`
# ================================================================
paths = os.listdir(exp_config.temporary_directory)
# dir_list element: (tmp_name, abs_path)
dir_list = []
for p in paths:
    abs_p = os.path.join(exp_config.temporary_directory, p)
    if p not in ['.autoPyTorch', 'smac3-output'] and not os.path.isfile(abs_p):
        dir_list.append((p, abs_p))

pathlib.Path(os.path.join(exp_config.temporary_directory, 'best_models')).mkdir(parents=True, exist_ok=True) 

for tmp_name, abs_path in dir_list:
    best_model = torch.load(os.path.join(abs_path, 'best.pth'), map_location='cpu')
    best_model.device = 'cpu'
    # Prepare input data for torchviz's backward propagation
    past_targets_shape = y_train[0].shape
    if len(past_targets_shape) == 1:
        past_targets_shape += (1, )
    past_features_shape = X_train[0].shape
    future_features_shape = X_test[0].shape
    past_targets = torch.randn(2, *past_targets_shape, device='cpu')
    future_targets = None
    past_features = torch.randn(2, *past_features_shape, device='cpu')
    future_features = torch.randn(2, *future_features_shape, device='cpu')
    past_observed_targets = torch.tensor([[1] * past_targets_shape[0]] * 2, dtype=torch.bool, device='cpu')
    best_model_output = best_model(past_targets, future_targets, past_features, future_features, past_observed_targets)
    # Draw with torchviz
    g = make_dot(best_model_output[0])
    g.render(filename=os.path.join(exp_config.temporary_directory, 'best_models/{}'.format(tmp_name)), format='png')


############################################################################
# Hint: Load prediction result
# =====================================================
# y_pred = np.loadtxt(exp_config.output_directory + "/y_pred.txt", delimiter=',')