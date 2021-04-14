from config import DATA_PATH, BATCH_SIZE, DIM, EXCLUDE_LAKE

from tensorflow.keras.preprocessing import image_dataset_from_directory

def data_split(val_split=0.2, seed=8):
    '''
    Method to generate train, validation and test image datasets

    Class arguments:
    val_split -- percentage of data from training directory to be used for validation
    seed -- int used to shuffle and split data for training/validation

    Returns:
    Tuple with three tf.data.Datasets for training, validation and test data, respectively
    '''

    train_dir = DATA_PATH + ('TrainingNoLake' if EXCLUDE_LAKE else 'Training')
    test_dir = DATA_PATH + 'Test'

    print(f'Using seed {seed}...')

    print('\nLoading training data...')
    train_data = image_dataset_from_directory(train_dir,
                                              label_mode='binary',
                                              batch_size=BATCH_SIZE,
                                              image_size=(DIM, DIM),
                                              validation_split=val_split,
                                              seed=seed,
                                              subset="training")

    print('\nLoading validation data...')
    val_data = image_dataset_from_directory(train_dir,
                                            label_mode='binary',
                                            batch_size=BATCH_SIZE,
                                            image_size=(DIM, DIM),
                                            validation_split=val_split,
                                            seed=seed,
                                            subset="validation")

    print('\nLoading test data...')
    test_data = image_dataset_from_directory(test_dir,
                                             label_mode='binary',
                                             batch_size=BATCH_SIZE,
                                             image_size=(DIM, DIM))

    return train_data, val_data, test_data