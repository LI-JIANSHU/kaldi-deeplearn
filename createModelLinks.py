import config
import os, subprocess

def run(sResultDir, sSrcDirs, sModelFileName):
    for i in xrange(0, config.PHONES):
        for j in xrange(i+1, config.PHONES):
            sDir = os.path.join(sResultDir, 'cp_%d_%d' % (i, j))
            if not os.path.exists(sDir):
                os.mkdir(sDir)
            
            sSrcModelPath = None
            for sSrcDir in sSrcDirs:
                s = os.path.join(sSrcDir, 'cp_%d_%d' % (i, j))
                if not os.path.exists(s):
                    continue
                s = os.path.join(s, sModelFileName)
                if os.path.exists(s):
                    sSrcModelPath = s
                    break
            if sSrcModelPath is None:
                print 'Failed to find the model for %d-%d' % (i, j)
                exit(1)
                
            args = ['ln', '-s', sSrcModelPath, os.path.join(sDir, 'model_BEST.fnn')]
            pr = subprocess.Popen(args, stderr=subprocess.STDOUT)
            pr.wait()
            if pr.returncode != 0:
                print 'Failed to create link for %d-%d' % (i, j)
                exit(1)
                
if __name__ == '__main__':
    run('/home/hvpham/kaldiPDNN/kaldi-trunk/egs/timit/s5/kaldi-deeplearn/models', \
        ['/home/hvpham/experiments/timit_conv_pairwise/cp', \
         '/home/hvpham/experiments/timit_conv_pairwise/cp/titan'], \
         'timit_conv_pairwise_train_BEST.fnn')
