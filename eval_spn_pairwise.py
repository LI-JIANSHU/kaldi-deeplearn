import util, deeplearn_pb2 as dl, config
import sys, os, subprocess
from kaldi_feat_io import KaldiReadIn, KaldiWriteOut
import numpy as np

def removeFile(sFile):
    try:
        os.remove(sFile)
    except Exception:
        pass
        
def createTestSet(ark_file, sDataFileTemplate, sDataDir, sDataFilePattern, sDataProtoFile):
    
    dbInfo = dl.DatabaseInfo()
    dbInfo.name = 'dataset_test'
    dbInfo.data_handler = 'deeplearn'
    dbInfo.path_prefix = sDataDir
    datasetInfo = dbInfo.data.add()
    datasetInfo.data_format = dl.DatasetInfo.PBM
    datasetInfo.size = 0
    datasetInfo.sparse_label = True
    datasetInfo.type = dl.DatasetInfo.TEST_SET
    datasetInfo.file_pattern = sDataFilePattern
    
    uttIDBatch = []
    uttIDLength = []
    featMatBatch = None
    batchSz = -1
    iFileIdx = 0
    kaldiIn = KaldiReadIn(ark_file)
    kaldiIn.open()
    uttID, featMat = kaldiIn.next()
    while featMat is not None:
        if batchSz < 0:
            batchSz = 400*1024*1024 / (4*featMat.shape[1])
            datasetInfo.dimensions = featMat.shape[1] + 1
            datasetInfo.label_start_index = datasetInfo.dimensions - 1
            
        if featMatBatch is None:
            featMatBatch = featMat
        else:
            featMatBatch = np.vstack([featMatBatch, featMat])
        uttIDBatch.append(uttID)
        uttIDLength.append(featMat.shape[0])
        
        if featMatBatch.shape[0] >= batchSz:
            util.WriteProto(sDataFileTemplate % iFileIdx, util.npy2ProtoMat(np.hstack([featMatBatch, np.zeros((featMatBatch.shape[0], 1))])))
            iFileIdx += 1
            datasetInfo.size += featMatBatch.shape[0]
            featMatBatch = None
        uttID, featMat = kaldiIn.next()
    kaldiIn.close()
    
    # last batch
    if featMatBatch is not None:
        util.WriteProto(sDataFileTemplate % iFileIdx, util.npy2ProtoMat(np.hstack([featMatBatch, np.zeros((featMatBatch.shape[0], 1))])))
        iFileIdx += 1
        datasetInfo.size += featMatBatch.shape[0]
    util.WriteProto(sDataProtoFile, dbInfo)
    return (uttIDBatch, uttIDLength)
    
def readPhonesToPdfIdMap():
    mRet = {}
    with open(config.PDFID_TO_PHONE_FILE, 'r') as f:
        for sLine in f:
            arr = sLine.split()
            if len(arr) == 0:
                continue
            mRet[int(arr[0])-1] = [int(x) for x in arr[1:]]
    return mRet
    
def computeMajorVotes(majorVotesHardFile, majorVotesSoftFile, resultDir):
    majorVotesHard = None
    majorVotesSoft = None
    phoneToPdfId = readPhonesToPdfIdMap()
    
    for i in xrange(0, config.PHONES - 1):
        for j in xrange(i + 1, config.PHONES):
            print 'Processing majorVotes %d %d\r' % (i, j),
            sys.stdout.flush()
            
            resultFile = os.path.join(resultDir, '%d_%d.csv' % (i, j))
            result = np.genfromtxt(resultFile, dtype=float, delimiter=',')
            
            if majorVotesHard is None:
                majorVotesHard = np.zeros((result.shape[0], 1 + config.CLASSES))
                majorVotesSoft = np.zeros((result.shape[0], 1 + config.CLASSES))
            assert majorVotesHard.shape[0] == result.shape[0] \
                and result.shape[1] == 2 + len(phoneToPdfId[i]) + len(phoneToPdfId[j])
            
            if np.sum(np.isnan(result[:, 2:])) > 0:
                print '\nFound NaN in', i, '-', j
                continue
                
            predicts = result[:, 1]
            
            for k, c in enumerate(phoneToPdfId[i]):
                majorVotesHard[predicts == k, 1+c] += 1
                majorVotesSoft[:, 1+c] += result[:, 2 + k]
            for k, c in enumerate(phoneToPdfId[j]):
                rc = k + len(phoneToPdfId[i])
                majorVotesHard[predicts == rc, 1+c] += 1
                majorVotesSoft[:, 1+c] += result[:, 2 + rc]
            
    majorVotesHard[:, 0] = majorVotesHard[:, 1:].argmax(1)
    majorVotesSoft[:, 0] = majorVotesSoft[:, 1:].argmax(1)
    np.savetxt(majorVotesHardFile, majorVotesHard, fmt='%f', delimiter=',')
    np.savetxt(majorVotesSoftFile, majorVotesSoft, fmt='%f', delimiter=',')
    return (majorVotesHard, majorVotesSoft)
    
################################################################################

if __name__ == '__main__':

    arguments = util.parse_arguments([x for x in sys.argv[1:]]) 

    if ((not arguments.has_key('ark_file')) 
        or (not arguments.has_key('wdir')) \
        or (not arguments.has_key('output_file_prefix')) \
        or (not arguments.has_key('weight_file')) \
        or (not arguments.has_key('model_files')) \
        or (not arguments.has_key('deeplearn_path'))):
        print "Error: the mandatory arguments --deeplearn-path --model-files --major-vote-prob --ark-file --weight-file --wdir --output-file-prefix"
        exit(1)
        
    # mandatory arguments
    ark_file = arguments['ark_file']
    wdir = os.path.abspath(arguments['wdir'])
    output_file_prefix = arguments['output_file_prefix']
    sWeightFile = arguments['weight_file']
    sModelFiles = arguments['model_files']
    sDeeplearnPath = arguments['deeplearn_path']
    
    # paths for output files
    output_scp = output_file_prefix + '.scp'
    output_ark = output_file_prefix + '.ark'
    removeFile(output_scp)
    removeFile(output_ark)
    
    # some sub-directory
    sDataDir = os.path.join(wdir, 'data')
    if not os.path.exists(sDataDir):
        os.mkdir(sDataDir)
    sResultDir = os.path.join(wdir, 'results')
    if not os.path.exists(sResultDir):
        os.mkdir(sResultDir)
    sMajorVoteDir = os.path.join(wdir, 'majorVote')
    if not os.path.exists(sMajorVoteDir):
        os.mkdir(sMajorVoteDir)
        
    # create the test set    
    sDataProtoFile = os.path.join(sDataDir, 'proto.pbtxt')
    (uttIDs, uttLens) = createTestSet(ark_file, os.path.join(sDataDir, 'test%05d.pbm'), \
                        sDataDir, 'test[0-9]*.pbm', sDataProtoFile)
    
    # prepare evalOp
    sEvalOpFile = os.path.join(sDataDir, 'eval.pbtxt')
    evalOp = dl.Operation()
    evalOp.name = 'test'
    evalOp.stop_condition.all_processed = True
    evalOp.operation_type = dl.Operation.TEST
    evalOp.data_proto = sDataProtoFile
    evalOp.randomize = False
    evalOp.get_last_piece = True
    evalOp.verbose = False
    
    # test on all models
    for i in xrange(0, config.PHONES):
        for j in xrange(i+1, config.PHONES):
            sResultFile = os.path.join(sResultDir, '%d_%d.csv' % (i, j))
            
            if os.path.exists(sResultFile):
                continue
            print 'Testing for %d-%d, writing results to %s' % (i, j, sResultFile)
            
            evalOp.result_file = sResultFile
            util.WriteProto(sEvalOpFile, evalOp)
            sModelFile = sModelFiles % (i, j)
            
            args = [sDeeplearnPath, 'eval', sModelFile, '--eval-op=%s' % sEvalOpFile]
            pr = subprocess.Popen(args, stderr=subprocess.STDOUT)
            pr.wait()
            if pr.returncode != 0:
                print 'Failed to test %d-%d' % (i, j)
                exit(1)
    
    # run majorityVote, compute "probabilities"
    sHardVoteFile = os.path.join(sMajorVoteDir, 'hard.csv')
    sSoftVoteFile = os.path.join(sMajorVoteDir, 'soft.csv')
    if os.path.exists(sHardVoteFile) and os.path.exists(sSoftVoteFile):
        majorVotesHard = np.genfromtxt(sHardVoteFile, delimiter=',')
        majorVotesSoft = np.genfromtxt(sSoftVoteFile, delimiter=',')
    else:
        (majorVotesHard, majorVotesSoft) = computeMajorVotes(sHardVoteFile, sSoftVoteFile, sResultDir)
    
    mProb = majorVotesSoft[:, 1:].copy()
    print mProb.sum(1)
    assert np.all(abs(mProb.sum(1) - config.CLASSIFIERS) < 1E-2)
    mProb /= mProb.sum(1)[:, None]

    kaldiOut = KaldiWriteOut(output_scp, output_ark)
    kaldiOut.open()
    rIdx = 0
    for i, uId in enumerate(uttIDs):
        kaldiOut.write(uId, mProb[rIdx:(rIdx + uttLens[i]), :])
        rIdx += uttLens[i]
    kaldiOut.close()
    
    # write dummy weights (identity matrix)
    with open(sWeightFile, 'wb') as fOut:
        w = np.eye(mProb.shape[1])
        (sz1, sz2) = w.shape
        fOut.write('<affinetransform> %d %d\n[\n' % (sz2, sz1))
        for c in xrange(0, sz2):
            fOut.write(' '.join([str(w[r, c]) for r in xrange(0, sz1)]) + '\n')
        fOut.write(']\n')
        
        # bias
        fOut.write('[ ')
        fOut.write(' '.join(['0.0' for i in xrange(0, sz1)]))
        fOut.write(' ]\n')
        fOut.write('<softmax> %d %d\n' % (sz2, sz1))


