#!/usr/bin/env python2

import sys, argparse, gzip, subprocess, random, datetime
from collections import deque
from util import read_reference, openz
import pdb
import numpy as np

def main(args):
    ref = read_reference(args.reference, args.verbose)
    o = openz(args.output, 'wt')

    #pdb.set_trace()
    hdr = openz(args.input).readline().strip().decode("utf-8").split('\t')
    n_cols = len(hdr)
    sys.stderr.write('{} columns in input\n'.format(n_cols))
    if hdr[0] == 'chrom':
        sys.stderr.write('Header present\n')
        #pdb.set_trace()
        o.write('{}\t{}\n'.format('\t'.join(hdr), args.label))
    else:
        sys.stderr.write('Header absent\n')
        hdr = None

    n_mut = 0
    then = datetime.datetime.now()
    with openz(args.input) as f:
        if hdr:
            f.readline()
        cchrom = None
        for s in f:
            v = s.strip().decode("utf-8").split('\t')
            chrom, pos = v[0], int(v[1])
            if cchrom != chrom:
                cchrom = chrom
                cpos = max(0, pos - args.window / 2)
                mpos = min(len(ref[chrom]) - 1, cpos + args.window)
                cpos = round(cpos)
                mpos = round(mpos)
                buf = deque(ref[chrom][cpos:mpos])
                gc = sum([1 for c in buf if c == 'C' or c == 'G'])
                at = sum([1 for c in buf if c == 'A' or c == 'T'])
            else:
                cpos2 = max(0, pos - args.window / 2)
                cdiff = cpos2 - cpos
                if cdiff > 0:
                    if cdiff < args.window:
                        # incremental update of buffer
                        for _ in range(round(cdiff)):
                            remove = buf.popleft()
                            if remove == 'C' or remove == 'G':
                                gc -= 1
                            elif remove == 'A' or remove == 'T':
                                at -= 1
                        insert = ref[cchrom][round(cpos+args.window):round(cpos+args.window+cdiff)]
                        gc += sum([1 for c in insert if c == 'C' or c == 'G'])
                        at += sum([1 for c in insert if c == 'A' or c == 'T'])
                        buf.extend(insert)
                    else:
                        # reinit buffer at cpos2
                        mpos = min(len(ref[chrom]) - 1, cpos2 + args.window)
                        buf = deque(ref[chrom][round(cpos2):round(mpos)])
                        gc = sum([1 for c in buf if c == 'C' or c == 'G'])
                        at = sum([1 for c in buf if c == 'A' or c == 'T'])
                    cpos = cpos2
            try:
                gc_ratio = 1.0 * gc / (gc + at)
            except:
                gc_ratio = 0
            
            o.write('{}\t{}\n'.format(s.strip().decode("utf-8"), gc_ratio))

            n_mut += 1
            if args.verbose and (n_mut % args.batch_size) == 0:
                now = datetime.datetime.now()
                td = now - then
                sys.stderr.write('{}: {} mutations. {}/mutation\n'.format(now, n_mut, td / args.batch_size))
                then = now

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('-n', '--window', default=1001, type=int)
    p.add_argument('-l', '--label', default='gc')
    p.add_argument('--batch-size', default=100000, type=int)
    p.add_argument('--reference', default='/scratch/pitkaene/tmp/hs37d5_1000GP.fa')
    p.add_argument('--verbose', action='store_true')
    main(p.parse_args())
