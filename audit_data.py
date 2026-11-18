import argparse
from tflmedsam.data import audit
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    audit(a.manifest,a.output,size=1024)
