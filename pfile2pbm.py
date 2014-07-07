from pfiles_io import read_dataset
import deeplearn_pb2 as dl
import numpy as np
import util, os
    
def pfile2Proto(pfilePath, filePrefix, pbmDir):
    pfile = read_dataset(pfilePath, {'partition': 1024*1024*400})
    
    dsInfo = dl.DatasetInfo()
    dsInfo.data_format = dl.DatasetInfo.PBM
    dsInfo.sparse_label = True
    dsInfo.file_pattern = '%s[0-9]*.pbm' % filePrefix
    dim = None
    sz = 0
    for i, (data, label) in enumerate(zip(pfile.feat_mats, pfile.label_vecs)):
        dataset = util.npy2ProtoMat(np.hstack([data, label[:, None]]))
        util.WriteProto(os.path.join(pbmDir, '%s%05d.pbm' % (filePrefix, i)), dataset)
        if dim is None:
            dim = data.shape[1] + 1
        if dim != data.shape[1] + 1:
            print dim, sz, data,shape, label.shape
        assert dim == data.shape[1] + 1
        sz += data.shape[0]
    dsInfo.size = sz
    dsInfo.dimensions = dim
    dsInfo.label_start_index = dim - 1
    return dsInfo
    
def createPbmDataset(pfiles, pbmDir, protoFilePath, gpuMem):
    assert gpuMem > 0.1
    
    dbInfo = dl.DatabaseInfo()
    
    for (name, sPath) in pfiles:
        assert name == 'train' or name == 'valid' or name == 'test'

        dsInfo = pfile2Proto(sPath, name + '_part', pbmDir)
        if name == 'train':
            dsInfo.type = dl.DatasetInfo.TRAIN_SET
        elif name == 'valid':
            dsInfo.type = dl.DatasetInfo.EVAL_SET
        else:
            dsInfo.type = dl.DatasetInfo.TEST_SET
        dbInfo.data.extend([dsInfo])
    
    dbInfo.name = 'dataset'
    dbInfo.data_handler = 'deeplearn'
    dbInfo.main_memory = 6.0
    dbInfo.gpu_memory = gpuMem
    dbInfo.path_prefix = pbmDir
    util.WriteProto(protoFilePath, dbInfo)
    
        
