import sys, os, subprocess, shutil
import pfile2pbm, util
import deeplearn_pb2 as dl

def createDir(sPath):
    if not os.path.exists(sPath):
        os.makedirs(sPath)
        
def weights2Kaldi(sOutFile, elements):
    with open(sOutFile, 'wb') as fOut:
        for e in elements:
            if type(e) is dl.EdgeData:
                w = e.weight
                assert w is not None
                sz2 = w.ld
                sz1 = len(w.data) / sz2
                fOut.write('<affinetransform> %d %d\n[\n' % (sz2, sz1))
                for c in xrange(0, sz2):
                    fOut.write(' '.join([str(w.data[r*sz2+c]) for r in xrange(0, sz1)]) + '\n')
                fOut.write(']\n')
            elif type(e) is dl.NodeData:
                b = e.bias
                assert b is not None and len(b.data) == b.ld
                fOut.write('[ %s ]\n' % ' '.join([str(x) for x in b.data]))
                sAct = ''
                if e.activation == dl.NodeData.SOFTMAX:
                    sAct = 'softmax'
                elif e.activation == dl.NodeData.TANH:
                    sAct = 'tanh'
                elif e.activation == dl.NodeData.LOGISTIC:
                    sAct = 'sigmoid'
                else:
                    assert False
                fOut.write('<%s> %d %d\n' % (sAct, b.ld, b.ld))
                
if __name__ == '__main__':
    
    arguments = util.parse_arguments([x for x in sys.argv[1:]]) 

    if (not arguments.has_key('train_data')) or (not arguments.has_key('valid_data')) \
        or (not arguments.has_key('num_outputs')) or (not arguments.has_key('wdir')) \
        or (not arguments.has_key('output_file')) or (not arguments.has_key('deeplearn_path')) \
        or (not arguments.has_key('weight_output_file')):
        print "Error: the mandatory arguments are: --weight-output-file --train-data --valid-data --num-outputs --wdir --output-file --deeplearn-path"
        exit(1)

    # mandatory arguments
    train_data_file = arguments['train_data']
    valid_data_file = arguments['valid_data']
    num_outputs = int(arguments['num_outputs'])
    wdir = os.path.abspath(arguments['wdir'])
    sOutputModelFile = arguments['output_file']
    sWeightOutputFile = arguments['weight_output_file']
    sDeeplearnPath = arguments['deeplearn_path']
    if 'gpu_mem' in arguments:
        fGpuMem = float(arguments['gpu_mem'])
    else:
        fGpuMem = 2.0
        
    # create dataset (from PFile to proto)
    sDataDir = os.path.join(wdir, 'data/proto')
    createDir(sDataDir)
    sDataProtoFile = os.path.join(sDataDir, 'data.pbtxt')
    if not os.path.exists(sDataProtoFile):
        pfile2pbm.createPbmDataset([['train', train_data_file], ['valid', valid_data_file]], sDataDir, sDataProtoFile, fGpuMem)
    else:
        print 'Found data.pbtxt at', sDataProtoFile
        print 'Skip generating dataset'
        sys.stdout.flush()
        
    if not os.path.exists(sOutputModelFile):    
        print 'modify the architecture...'
        sModelDir = os.path.join(wdir, 'model/')
        createDir(sModelDir)
        sCurrentDir = os.path.split(os.path.realpath(os.path.abspath(__file__)))[0]
        model = util.ReadProto(os.path.join(sCurrentDir, 'prototype/conv_timit.pbtxt'), dl.ModelData())
        model.name = 'spn_conv'
        for n in model.nodes:
            if n.name == 'output':
                n.dimension = num_outputs
                break

	# added by LJS: Add two more hidden layers for the CNN for SWDB 
        for n in model.nodes:
            if n.name == 'hidden1':
                tmp=n
                break
	
	model.nodes.extend([tmp,tmp])
	model.nodes[-2].name='hidden3'	
	model.nodes[-1].name='hidden4'	

	# added by LJS: Add the edges for the two additional layers: 
        for idx in range(len(model.edges)):
	    e=model.edges[idx]
            if ((e.node1 == 'hidden1') and (e.node2 == 'hidden2')):
                break

	model.edges.extend([e,e])

	model.edges[idx+1].node1='hidden2'
	model.edges[idx+1].node2='hidden3'
	model.edges[idx+2].node1='hidden3'
	model.edges[idx+2].node2='hidden4'
	model.edges[idx+3].node1='hidden4'
	model.edges[idx+3].node2='output'

	# added by LJS: Change tanh to sigmoid 
	if True:
        	for n in model.nodes:
			if n.name=='input' or n.name=='output':
				continue
			n.activation=2 #2 is the code for LOGISTIC
	
	# add by ljs: use pretrained option from CNN
	if True:
		model.hyper_params.initialization=5 # 5 is for PRETRAINED
		for e in model.edges:
			if e.node1=='input':
				e.hyper_params.initialization=5 # 5 is for PRETRAINED
				e.hyper_params.pretrained_init='/home/jsli/spn_initial/input_conv1.csv'	
			if e.node1=='conv1':
				e.hyper_params.initialization=5 # 5 is for PRETRAINED
				e.hyper_params.pretrained_init='/home/jsli/spn_initial/conv1_conv2.csv'
			if e.node1=='conv2':
				e.hyper_params.initialization=5 # 5 is for PRETRAINED
				e.hyper_params.pretrained_init='/home/jsli/spn_initial/conv2_h1.csv'
			if e.node1=='hidden1':
				e.hyper_params.initialization=5 # 5 is for PRETRAINED
				e.hyper_params.pretrained_init='/home/jsli/spn_initial/h1_h2.csv'
			if e.node1=='hidden2':
				e.hyper_params.initialization=5 # 5 is for PRETRAINED
				e.hyper_params.pretrained_init='/home/jsli/spn_initial/h2_h3.csv'
			if e.node1=='hidden3':
				e.hyper_params.initialization=5 # 5 is for PRETRAINED
				e.hyper_params.pretrained_init='/home/jsli/spn_initial/h3_h4.csv'
			if e.node1=='hidden4':
				e.hyper_params.initialization=5 # 5 is for PRETRAINED
				e.hyper_params.pretrained_init='/home/jsli/spn_initial/h4_output.csv'
		for n in model.nodes:
			if n.name=='conv1':
				n.hyper_params.initialization=5 # 5 is for PRETRAINED
				n.hyper_params.pretrained_init='/home/jsli/spn_initial/conv1_bias.csv'	
			if n.name=='conv2':
				n.hyper_params.initialization=5 # 5 is for PRETRAINED
				n.hyper_params.pretrained_init='/home/jsli/spn_initial/conv2_bias.csv'	
			if n.name=='output':
				n.hyper_params.initialization=5 # 5 is for PRETRAINED
				n.hyper_params.pretrained_init='/home/jsli/spn_initial/output_bias.csv'	
			if n.name=='hidden1':
				n.hyper_params.initialization=5 # 5 is for PRETRAINED
				n.hyper_params.pretrained_init='/home/jsli/spn_initial/h1_bias.csv'	
			if n.name=='hidden2':
				n.hyper_params.initialization=5 # 5 is for PRETRAINED
				n.hyper_params.pretrained_init='/home/jsli/spn_initial/h2_bias.csv'	
			if n.name=='hidden3':
				n.hyper_params.initialization=5 # 5 is for PRETRAINED
				n.hyper_params.pretrained_init='/home/jsli/spn_initial/h3_bias.csv'	
			if n.name=='hidden4':
				n.hyper_params.initialization=5 # 5 is for PRETRAINED
				n.hyper_params.pretrained_init='/home/jsli/spn_initial/h4_bias.csv'	

        # modify training parameters to those of CNN
	if True:   
		model.hyper_params.base_learningrate=0.08
		model.hyper_params.learningrate_decay_half_life=10000
		model.hyper_params.initial_momentum=0.5
		model.hyper_params.final_momentum=0.5
		model.hyper_params.apply_l2_decay=False
		model.hyper_params.min_learningrate = 0.00001


        sModelFile = os.path.join(sModelDir, 'spn_conv.pbtxt')
        util.WriteProto(sModelFile, model)
	
        trainOp = util.ReadProto(os.path.join(sCurrentDir, 'prototype/train.pbtxt'), dl.Operation())
        sCheckpointDir = os.path.join(sModelDir, 'cp')
        trainOp.name = 'train'
        trainOp.data_proto = sDataProtoFile
        trainOp.checkpoint_directory = sCheckpointDir
        trainOp.verbose = False
	trainOp.batch_size=256
	trainOp.stop_condition.steps=50000
	trainOp.checkpoint_after=2000
        sTrainOpFile = os.path.join(sModelDir, 'train.pbtxt')
        util.WriteProto(sTrainOpFile, trainOp)
        
        evalOp = util.ReadProto(os.path.join(sCurrentDir, 'prototype/eval.pbtxt'), dl.Operation())
        evalOp.data_proto = sDataProtoFile
        evalOp.verbose = False
        evalOp.result_file = os.path.join(sModelDir, 'result_test.csv')
        evalOp.result_file_validation_set = os.path.join(sModelDir, 'result_val.csv')
	evalOp.batch_size=256
        sEvalOpFile = os.path.join(sModelDir, 'eval.pbtxt')
        util.WriteProto(sEvalOpFile, evalOp)
        print 'Model files are saved.'
        print 'Run the model...'        
        # run it
        args = [sDeeplearnPath, 'train', sModelFile, \
                '--train-op=%s' % sTrainOpFile,
                '--eval-op=%s' % sEvalOpFile]
        pr = subprocess.Popen(args, stderr=subprocess.STDOUT)
        pr.wait()
        if pr.returncode != 0:
            exit(1)
        
        print 'write the output..'
        sBestModel = os.path.join(sCheckpointDir, 'spn_conv_train_BEST.fnn')
        if not os.path.exists(sBestModel):
            sBestModel = os.path.join(sCheckpointDir, 'spn_conv_train_LAST.fnn')
        if not os.path.exists(sBestModel):
            print "Couldn't find the best SPN model. Terminating..."
            exit(1)
        shutil.copy(sBestModel, sOutputModelFile)
    else:
        print 'Found the trained model at', sOutputModelFile
        print 'Not training it again.'
        sys.stdout.flush()
        
    bestModel = util.ReadProto(sOutputModelFile, dl.ModelData())
    exportedElements = []
    sLayerName = 'conv2'
    while 1:
        ee = [e for e in bestModel.edges if e.node1 == sLayerName]
        if len(ee) == 0:
            break
        else:
            sLayerName = ee[0].node2
            exportedElements.append(ee[0])
            exportedElements.extend([n for n in bestModel.nodes if n.name == sLayerName])
    weights2Kaldi(sWeightOutputFile, exportedElements)
    
    
    
