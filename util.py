import os, deeplearn_pb2 as dl
from google.protobuf import text_format
import numpy as np

def parse_arguments(arg_elements):
    args = {}
    arg_num = len(arg_elements) / 2
    for i in xrange(arg_num):
        key = arg_elements[2*i].replace("--","").replace("-", "_");
        args[key] = arg_elements[2*i+1]
    return args
    
################################################################################

def ReadProto(fPath, proto):
    if os.path.splitext(fPath)[1] == '.pbtxt':
        proto_pbtxt = open(fPath, 'r')
        text_format.Merge(proto_pbtxt.read(), proto)
    else:
        f = open(fPath, 'rb')
        proto.ParseFromString(f.read())
        f.close()
    return proto

def WriteProto(fPath, proto):
  if os.path.splitext(fPath)[1] == '.pbtxt':
    with open(fPath, 'w') as f:
      text_format.PrintMessage(proto, f)
  else:
    f = open(fPath, 'wb')
    f.write(proto.SerializeToString())
    f.close()

################################################################################

def npy2ProtoMat(npyMat):
    assert len(npyMat.shape) <= 2

    pbMat = dl.Matrix()
    if len(npyMat.shape) == 0:
        pbMat.ld = 0
    elif len(npyMat.shape) == 1:
        # vector
        pbMat.ld = 1
        pbMat.data.extend(npyMat.tolist())
    else:
        # matrix
        pbMat.ld = npyMat.shape[1]
        pbMat.data.extend(np.squeeze(np.reshape(npyMat, [1, -1])).tolist())
    return pbMat
    
def proto2Npy(pbMat):
    assert pbMat.ld >= 0

    if pbMat.ld == 0:
        assert len(pbMat.data) == 0
        return np.zeros([])
    
    nElem = len(pbMat.data)
    nCols = pbMat.ld
    nRows = nElem / nCols
    assert nElem % nCols == 0
    npyMat = np.zeros([nRows, nCols])
    for r in xrange(0, nRows):
        npyMat[r, :] = pbMat.data[(nCols*r):(nCols*(r+1))]
    return npyMat
