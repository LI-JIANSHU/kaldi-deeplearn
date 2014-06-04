import sys, os, subprocess
from kaldi_feat_io import KaldiReadIn, KaldiWriteOut
import deeplearn_pb2 as dl, util, glob
import numpy as np

def extractRepresentation(data, wdir, sDeeplearnPath, sModelFile):
    # append label column
    data = np.hstack([data, np.zeros((data.shape[0], 1))])
    npyData = util.npy2ProtoMat(data)
    sDataFile = os.path.join(wdir, 'input.pbm')
    util.WriteProto(sDataFile, npyData)

    sDataProtoFile = os.path.join(wdir, 'data.pbtxt')
    dbInfo = dl.DatabaseInfo()
    dbInfo.name = 'dataset_extract'
    dbInfo.data_handler = 'deeplearn'
    dbInfo.path_prefix = wdir
    datasetInfo = dbInfo.data.add()
    datasetInfo.data_format = dl.DatasetInfo.PBM
    datasetInfo.size = data.shape[0]
    datasetInfo.dimensions = data.shape[1]
    datasetInfo.label_start_index = datasetInfo.dimensions - 1
    datasetInfo.sparse_label = True
    datasetInfo.type = dl.DatasetInfo.TEST_SET
    datasetInfo.file_pattern = 'input.pbm'
    util.WriteProto(sDataProtoFile, dbInfo)
    
    sEvalOpFile = os.path.join(wdir, 'eval.pbtxt')
    sExtractedActs = os.path.join(wdir, 'acts')
    sLayerName = 'conv2'
    if not os.path.exists(sExtractedActs):
        os.mkdir(sExtractedActs)
    evalOp = dl.Operation()
    evalOp.name = 'extract'
    evalOp.stop_condition.all_processed = True
    evalOp.operation_type = dl.Operation.TEST
    evalOp.data_proto = sDataProtoFile
    evalOp.randomize = False
    evalOp.get_last_piece = True
    evalOp.verbose = False
    evalOp.extracted_layers.append(sLayerName)
    evalOp.extracted_output_dir = sExtractedActs
    evalOp.extracted_data_format = dl.DatasetInfo.PBM
    evalOp.extracted_data_sets.append(dl.DatasetInfo.TEST_SET)
    util.WriteProto(sEvalOpFile, evalOp)
    
    sOutFileTemplate = os.path.join(sExtractedActs, sLayerName, '*.pbm')
    for s in sorted(glob.glob(sOutFileTemplate)):
        try:
            os.remove(s)
        except Exception:
            pass
            
    # run the network...
    args = [sDeeplearnPath, 'extract', sModelFile, '--eval-op=%s' % sEvalOpFile]
    pr = subprocess.Popen(args, stderr=subprocess.STDOUT)
    pr.wait()
    if pr.returncode != 0:
        print 'Failed to extract representations'
        exit(1)
    
    # read Dataset
    mOutput = None
    for s in sorted(glob.glob(sOutFileTemplate)):
        m = util.proto2Npy(util.ReadProto(s, dl.Matrix()))
        if mOutput is None:
            mOutput = m
        else:
            mOutput = np.vstack([mOutput, m])
            
    if mOutput.shape[0] != data.shape[0]:
        print 'Invalid results'
        exit(1)
    return mOutput
    
def removeFile(sFile):
    try:
        os.remove(sFile)
    except Exception:
        pass
        
if __name__ == '__main__':

    arguments = util.parse_arguments([x for x in sys.argv[1:]]) 

    if (not arguments.has_key('ark_file')) or (not arguments.has_key('wdir')) \
        or (not arguments.has_key('output_file_prefix')) or (not arguments.has_key('model_file')) \
        or (not arguments.has_key('deeplearn_path')):
        print "Error: the mandatory arguments are: --deeplearn-path --ark-file --wdir --output-file-prefix --model-file"
        exit(1)

    # mandatory arguments
    ark_file = arguments['ark_file']
    wdir = os.path.abspath(arguments['wdir'])
    output_file_prefix = arguments['output_file_prefix']
    sModelFile = arguments['model_file']
    sDeeplearnPath = arguments['deeplearn_path']
    
    # paths for output files
    output_scp = output_file_prefix + '.scp'
    output_ark = output_file_prefix + '.ark'
    removeFile(output_scp)
    removeFile(output_ark)
    
    sDataDir = os.path.join(wdir, 'data')
    if not os.path.exists(sDataDir):
        os.mkdir(sDataDir)
    
    kaldiIn = KaldiReadIn(ark_file)
    kaldiIn.open()
    kaldiOut = KaldiWriteOut(output_scp,output_ark)
    kaldiOut.open()
    uttIDBatch = []
    uttIDLength = []
    featMatBatch = None
    batchSz = -1
    uttID, featMat = kaldiIn.next()
    while featMat is not None:
        if batchSz < 0:
            batchSz = 300*1024*1024 / (4*featMat.shape[1])
            
        if featMatBatch is None:
            featMatBatch = featMat
        else:
            featMatBatch = np.vstack([featMatBatch, featMat])
        uttIDBatch.append(uttID)
        uttIDLength.append(featMat.shape[0])
        
        if featMatBatch.shape[0] >= batchSz:
            featOut = extractRepresentation(featMatBatch, sDataDir, sDeeplearnPath, sModelFile)
            rIdx = 0
            for i, uId in enumerate(uttIDBatch):
                kaldiOut.write(uId, featOut[rIdx:(rIdx + uttIDLength[i]), :])
                rIdx += uttIDLength[i]
            
            featMatBatch = None
            uttIDBatch = []
            uttIDLength = []
        uttID, featMat = kaldiIn.next()
    
    # final batch
    if featMatBatch.shape[0] > 0:
        featOut = extractRepresentation(featMatBatch, sDataDir, sDeeplearnPath, sModelFile)
        rIdx = 0
        for i, uId in enumerate(uttIDBatch):
            kaldiOut.write(uId, featOut[rIdx:(rIdx + uttIDLength[i]), :])
            rIdx += uttIDLength[i]
    kaldiIn.close()
    kaldiOut.close()




