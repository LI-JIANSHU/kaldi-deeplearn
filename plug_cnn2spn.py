import os,sys
import util
import deeplearn_pb2 as dl
from train_spn import weights2Kaldi


if len(sys.argv) != 4:
	print "Error!!!  Usage:" +sys.argv[0]+" path/to/dir_containing_tmp_files path/to/spn_conv path/to/new_spn_conv" 
	exit(1)

sdir = sys.argv[1]
spn = sys.argv[2]
spn_new=sys.argv[3]

print 'Loading SPN model...'
md = util.ReadProto(spn, dl.ModelData())

print 'Modifying the edges...'
for e in md.edges:
	if e.node1=='input':
		print 'Processing conv1...'
		fh=open(sdir+'/input_conv1.weight')
		data=fh.readline().split()
		fh.close()
		assert len(e.weight.data)==len(data)	
		for i in range(len(data)):
			e.weight.data[i]=float(data[i])
	elif e.node1=='conv1':
		print 'Processing conv2...'
		fh=open(sdir+'/conv1_conv2.weight')
		data=fh.readline().split()
		fh.close()
		assert len(e.weight.data)==len(data)	
		for i in range(len(data)):
			e.weight.data[i]=float(data[i])
	elif e.node1=='conv2':
		print 'Processing h1...'
		fh=open(sdir+'/conv2_h1.weight')
		for idx in range(e.weight.ld):
			data=fh.readline().split()
			for idx2 in range(len(data)):
				e.weight.data[idx2*e.weight.ld+idx]=float(data[idx2])			
		fh.close()
	elif e.node1=='hidden1':
		print 'Processing h2...'
		fh=open(sdir+'/h1_h2.weight')
		for idx in range(e.weight.ld):
			data=fh.readline().split()
			for idx2 in range(len(data)):
				e.weight.data[idx2*e.weight.ld+idx]=float(data[idx2])	
		fh.close()
	elif e.node1=='hidden2':
		print 'Processing h3...'
		fh=open(sdir+'/h2_h3.weight')
		for idx in range(e.weight.ld):
			data=fh.readline().split()
			for idx2 in range(len(data)):
				e.weight.data[idx2*e.weight.ld+idx]=float(data[idx2])
		fh.close()
	elif e.node1=='hidden3':
		print 'Processing h4...'
		fh=open(sdir+'/h3_h4.weight')
		for idx in range(e.weight.ld):
			data=fh.readline().split()
			for idx2 in range(len(data)):
				e.weight.data[idx2*e.weight.ld+idx]=float(data[idx2])	
		fh.close()
	elif e.node1=='hidden4':
		print 'Processing output...'
		fh=open(sdir+'/h4_output.weight')
		for idx in range(e.weight.ld):
			data=fh.readline().split()
			for idx2 in range(len(data)):
				e.weight.data[idx2*e.weight.ld+idx]=float(data[idx2])
		fh.close()
	else:
		print "Error in weights..."
		exit(1)

print 'Done.'
print 'Modifying the bias...'

for n in md.nodes:
	if n.name=='conv1':
		print 'Processing conv1...'
		fh=open(sdir+'/conv1.bias')
		data=fh.readline().split()
		fh.close()
		assert len(data)==64
		for idx in range(64):
			val=float(data[idx])
			for idx2 in range(13*13):
				n.bias.data[idx*13*13+idx2]=val
	elif n.name=='conv2':
		print 'Processing conv2...'
		fh=open(sdir+'/conv2.bias')
		data=fh.readline().split()
		fh.close()
		assert len(data)==128
		for idx in range(128):
			val=float(data[idx])
			for idx2 in range(9):
				n.bias.data[idx*9+idx2]=val
	elif n.name=='hidden1':
		print 'Processing h1...'
		fh=open(sdir+'/h1.bias')
		data=fh.readline().split()
		fh.close()
		assert len(data)==len(n.bias.data)
		for idx in range(len(data)):
			n.bias.data[idx]=float(data[idx])
	elif n.name=='hidden2':
		print 'Processing h2...'
		fh=open(sdir+'/h2.bias')
		data=fh.readline().split()
		fh.close()
		assert len(data)==len(n.bias.data)
		for idx in range(len(data)):
			n.bias.data[idx]=float(data[idx])
	elif n.name=='hidden3':
		print 'Processing h3...'
		fh=open(sdir+'/h3.bias')
		data=fh.readline().split()
		fh.close()
		assert len(data)==len(n.bias.data)
		for idx in range(len(data)):
			n.bias.data[idx]=float(data[idx])
	elif n.name=='hidden4':
		print 'Processing h4...'
		fh=open(sdir+'/h4.bias')
		data=fh.readline().split()
		fh.close()
		assert len(data)==len(n.bias.data)
		for idx in range(len(data)):
			n.bias.data[idx]=float(data[idx])
	elif n.name=='output':
		print 'Processing output...'
		fh=open(sdir+'/output.bias')
		data=fh.readline().split()
		fh.close()
		assert len(data)==len(n.bias.data)
		for idx in range(len(data)):
			n.bias.data[idx]=float(data[idx])
	elif n.name=='input':
		print 'There is no bias for input'
	else:
		print "Error in bias."
		exit(1)
print 'Done'

print 'Changing activation to sigmoid...'
for n in md.nodes:
	if n.name=='conv1' or n.name=='conv2' or n.name=='hidden1' or n.name=='hidden2' or n.name=='hidden3' or n.name=='hidden4':
		n.activation=2  #2 is LOGISTIC


print 'Saving the model...'
util.WriteProto(spn_new,md)
print 'Modified SPN is saved to '+spn_new


# Regenerate the dnn.nnet file. It should be identical to dnn.nnet in CNN
#exportedElements = []
#sLayerName = 'conv2'
#while 1:
#	ee = [e for e in md.edges if e.node1 == sLayerName]
#        if len(ee) == 0:
#            break
#        else:
#            sLayerName = ee[0].node2
#            exportedElements.append(ee[0])
#            exportedElements.extend([n for n in md.nodes if n.name == sLayerName])

#dnn='test_dnn'
#weights2Kaldi(dnn, exportedElements)
#print 'New dnn.nnet file is saved to '+dnn

