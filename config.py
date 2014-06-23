DATA_DIR = '/home/hvpham/experiments/data/timit_conv/'
DATA_FILE_TEMPLATES = [DATA_DIR + 'train_part*.pbm', DATA_DIR + 'valid_part*.pbm']      # train, validation and test (if any)
DATA_FILE_NUMPY = [DATA_DIR + 'train.npy', DATA_DIR + 'valid.npy']      # train, validation and test (if any)
PAIRWISE_DATA_DIR = '/home/hvpham/experiments/data/timit_conv_pairwise/'
PAIRWISE_PBM_DIR = PAIRWISE_DATA_DIR + 'pbm/'
PAIRWISE_PROTO_DIR = PAIRWISE_DATA_DIR + 'proto/'
PAIRWISE_PROTO_TEMPLATE = PAIRWISE_PROTO_DIR + '%d_%d.pbtxt'

import os

PDFID_TO_PHONE_FILE = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'pdfToPhone.txt')
ROW_INDICES_FILE = 'rowIndices.pkl'

EXPE_DIR = '/home/hvpham/experiments/timit_conv_pairwise/'

DEEPLEARN_PATH = '/home/hvpham/experiments/bin/deeplearn'

DELETE_DATA_FILES = True

CLASSES = 1957
PHONES = 39
CLASSIFIERS = 741           # PHONES choose 2
