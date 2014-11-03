import os,sys
import util
import deeplearn_pb2 as dl
from train_spn import weights2Kaldi


if len(sys.argv) != 3:
	print "Error!!!  Usage:" +sys.argv[0]+" path/to/spn_conv.fnn path/of/folder/to designated results" 
	exit(1)

spn = sys.argv[1]
rdir=sys.argv[2]

print 'Loading SPN model...'
md = util.ReadProto( os.path.abspath(spn), dl.ModelData())

print 'Writing the edges...'
for e in md.edges:
	if e.node1=='input':
		print 'Processing conv1...'
		assert len(e.weight.data)%e.weight.ld==0
		sz2=e.weight.ld
		sz1=len(e.weight.data)/e.weight.ld
		fh=open(rdir+'/input_conv1.csv','wb')
		for r in xrange(0, sz1):
			fh.write(','.join([str(e.weight.data[sz2*r+c]) for c in xrange(0, sz2)]) + '\n')  
		fh.close()
		
	elif e.node1=='conv1':
		print 'Processing conv2...'
		assert len(e.weight.data)%e.weight.ld==0
		sz2=e.weight.ld
		sz1=len(e.weight.data)/e.weight.ld
		fh=open(rdir+'/conv1_conv2.csv','wb')
		for r in xrange(0, sz1):
			fh.write(','.join([str(e.weight.data[sz2*r+c]) for c in xrange(0, sz2)]) + '\n')  
		fh.close()

	elif e.node1=='conv2':
		print 'Processing h1...'
		assert len(e.weight.data)%e.weight.ld==0
		sz2=e.weight.ld
		sz1=len(e.weight.data)/e.weight.ld
		fh=open(rdir+'/conv2_h1.csv','wb')
		for r in xrange(0, sz1):
			fh.write(','.join([str(e.weight.data[sz2*r+c]) for c in xrange(0, sz2)]) + '\n')  
		fh.close()


	elif e.node1=='hidden1':
		print 'Processing h2...'
		assert len(e.weight.data)%e.weight.ld==0
		sz2=e.weight.ld
		sz1=len(e.weight.data)/e.weight.ld
		fh=open(rdir+'/h1_h2.csv','wb')
		for r in xrange(0, sz1):
			fh.write(','.join([str(e.weight.data[sz2*r+c]) for c in xrange(0, sz2)]) + '\n')  
		fh.close()

	elif e.node1=='hidden2':
		print 'Processing h3...'
		assert len(e.weight.data)%e.weight.ld==0
		sz2=e.weight.ld
		sz1=len(e.weight.data)/e.weight.ld
		fh=open(rdir+'/h2_h3.csv','wb')
		for r in xrange(0, sz1):
			fh.write(','.join([str(e.weight.data[sz2*r+c]) for c in xrange(0, sz2)]) + '\n')  
		fh.close()

	elif e.node1=='hidden3':
		print 'Processing h4...'
		assert len(e.weight.data)%e.weight.ld==0
		sz2=e.weight.ld
		sz1=len(e.weight.data)/e.weight.ld
		fh=open(rdir+'/h3_h4.csv','wb')
		for r in xrange(0, sz1):
			fh.write(','.join([str(e.weight.data[sz2*r+c]) for c in xrange(0, sz2)]) + '\n')  
		fh.close()

	elif e.node1=='hidden4':
		print 'Processing output...'
		assert len(e.weight.data)%e.weight.ld==0
		sz2=e.weight.ld
		sz1=len(e.weight.data)/e.weight.ld
		fh=open(rdir+'/h4_output.csv','wb')
		for r in xrange(0, sz1):
			fh.write(','.join([str(e.weight.data[sz2*r+c]) for c in xrange(0, sz2)]) + '\n')  
		fh.close()

	else:
		print "Error in weights..."
		exit(1)

print 'Done.'
print 'Writing the bias...'

for n in md.nodes:
	if n.name=='conv1':
		print 'Processing conv1...'
		fh=open(rdir+'/conv1_bias.csv','wb')
		fh.write(','.join([str(n.bias.data[c]) for c in xrange(0, n.bias.ld)]) + '\n')  

	elif n.name=='conv2':
		print 'Processing conv2...'
		fh=open(rdir+'/conv2_bias.csv','wb')
		fh.write(','.join([str(n.bias.data[c]) for c in xrange(0, n.bias.ld)]) + '\n')  

	elif n.name=='hidden1':
		print 'Processing h1...'
		fh=open(rdir+'/h1_bias.csv','wb')
		fh.write(','.join([str(n.bias.data[c]) for c in xrange(0, n.bias.ld)]) + '\n')  

	elif n.name=='hidden2':
		print 'Processing h2...'
		fh=open(rdir+'/h2_bias.csv','wb')
		fh.write(','.join([str(n.bias.data[c]) for c in xrange(0, n.bias.ld)]) + '\n')  

	elif n.name=='hidden3':
		print 'Processing h3...'
		fh=open(rdir+'/h3_bias.csv','wb')
		fh.write(','.join([str(n.bias.data[c]) for c in xrange(0, n.bias.ld)]) + '\n')  
	elif n.name=='hidden4':
		print 'Processing h4...'
		fh=open(rdir+'/h4_bias.csv','wb')
		fh.write(','.join([str(n.bias.data[c]) for c in xrange(0, n.bias.ld)]) + '\n')  
	elif n.name=='output':
		print 'Processing output...'
		fh=open(rdir+'/output_bias.csv','wb')
		fh.write(','.join([str(n.bias.data[c]) for c in xrange(0, n.bias.ld)]) + '\n')  
	elif n.name=='input':
		print 'There is no bias for input'
	else:
		print "Error in bias."
		exit(1)
print 'Done'

print 'The resulting csv files are in '+rdir
