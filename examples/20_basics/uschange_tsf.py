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
import warnings
import copy
import numpy as np
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sktime.datasets import load_uschange
targets, features = load_uschange()

exp_index = 'test'
temporary_directory = './uschange/uschange_tmp_{}'.format(exp_index)
output_directory = './uschange/uschange_out_{}'.format(exp_index)
forecasting_horizon = 10

func_eval_time_limit_secs = int(sys.argv[1]) * 60
total_walltime_limit = int(sys.argv[2]) * 60
print('args -- func_eval_time_limit_secs:{}, total_walltime_limit:{}'.format(func_eval_time_limit_secs, total_walltime_limit))

# Dataset optimized by APT-TS can be a list of np.ndarray / pd.DataFrame where each series represents an element in the
# list, or a single pd.DataFrame that records the series
# index information: to which series the timestep belongs? This id can be stored as the DataFrame's index or a separate
# column
# Within each series, we take the last forecasting_horizon as test targets. The items before that as training targets
# Normally the value to be forecasted should follow the training sets
y_train = [targets[: -forecasting_horizon]]
y_test = [targets[-forecasting_horizon:]]

# same for features. For uni-variant models, X_train, X_test can be omitted and set as None
X_train = [features[: -forecasting_horizon]]
# Here x_test indicates the 'known future features': they are the features known previously, features that are unknown
# could be replaced with NAN or zeros (which will not be used by our networks). If no feature is known beforehand,
# we could also omit X_test
known_future_features = list(features.columns)
X_test = [features[-forecasting_horizon:]]

# start_times = [targets.index.to_timestamp()[0]]
# freq = '1Y'

print("data ok")

from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
############################################################################
# Build and fit a forecaster
# ==========================
api = TimeSeriesForecastingTask(
    temporary_directory=temporary_directory,
    output_directory=output_directory,
    delete_tmp_folder_after_terminate=False,
    delete_output_folder_after_terminate=False,
)

############################################################################
# Search for an ensemble of machine learning algorithms
# =====================================================
api.search(
    X_train=X_train,
    y_train=copy.deepcopy(y_train),
    X_test=X_test,
    optimize_metric='mean_MASE_forecasting',
    n_prediction_steps=forecasting_horizon,
    memory_limit=16 * 1024,  # Currently, forecasting models use much more memories
    # freq=freq,
    # start_times=start_times,
    func_eval_time_limit_secs=func_eval_time_limit_secs,
    total_walltime_limit=total_walltime_limit,
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

y_pred = api.predict(test_sets)
np.savetxt(output_directory + "/y_pred.txt", y_pred, delimiter=',')
# print('== socre :', api.score(y_pred, y_test, XXXX, forecasting_horizon))
print('== stat  :', api.sprint_statistics())
# print('== models:', api.show_models())