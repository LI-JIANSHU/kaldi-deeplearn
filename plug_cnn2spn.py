import os,sys
import util
import deeplearn_pb2 as dl


if len(sys.argv) != 5:
	print "Error!!!  Usage:" +sys.argv[0]+" path/to/cnn_layer1 path/to/cnn_layer2 path/to/spn_conv path/to/new_spn_conv" 
	exit(1)

layer1 = sys.argv[1]
layer2 = sys.argv[2]
spn = sys.argv[3]
spn_new=sys.argv[4]

print 'Loading SPN model...'
md = util.ReadProto(spn, dl.ModelData())
for i in range(len(md.edges)):
	if md.edges[i].node1=='input' and md.edges[i].node2=='conv1':
		idx1=i
	if md.edges[i].node1=='conv1' and md.edges[i].node2=='conv2':
		idx2=i
	


print 'Loading CNN layer1 data...'
fh=open(layer1)
c1=fh.readline().split()

print 'Loading CNN layer2 data...'
fh=open(layer2)
c2=fh.readline().split()

print 'Modifying SPN...'
assert len(c1)==len(md.edges[idx1].weight.data)
assert len(c2)==len(md.edges[idx2].weight.data)
for i in range(len(c1)):
	md.edges[idx1].weight.data[i]=float(c1[i])
for i in range(len(c2)):
	md.edges[idx2].weight.data[i]=float(c2[i])


util.WriteProto(spn_new,md)
print 'Modified SPN is saved to '+spn_new
