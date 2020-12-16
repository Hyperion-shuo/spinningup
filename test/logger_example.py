from spinup.utils.logx import EpochLogger

epoch_logger = EpochLogger(output_dir='../data/logger_example')
for i in range(10):
    epoch_logger.store(Test=i)
epoch_logger.log_tabular('Test', with_min_and_max=True)
epoch_logger.dump_tabular()