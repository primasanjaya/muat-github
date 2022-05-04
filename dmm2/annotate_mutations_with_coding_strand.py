#!/usr/bin/env python2
'''
Annotate a mutation TSV with coding strand information.

Adds a column to the input with the following possible values:
    +   Pyrimidine reference base of the mutation is on the plus strand
    -   Pyrimidine reference base of the mutation is on the minus strand
    ?   Mutation overlaps both plus and minus stranded exons
    =   Mutation does not overlap an exon
'''

import sys, argparse, gzip, subprocess, random, datetime
from util import read_reference, openz

def main(args):
    o = openz(args.output, 'w')
    hdr = openz(args.input).readline().strip().split('\t')
    n_cols = len(hdr)
    sys.stderr.write('{} columns in input\n'.format(n_cols))
    if hdr[0] == 'chrom':
        sys.stderr.write('Header present\n')
        o.write('{}\tstrand\n'.format('\t'.join(hdr)))
    else:
        sys.stderr.write('Header absent\n')
        hdr = None
    sys.stderr.write('Reading reference: ')
    reference = read_reference(args.ref, verbose=True)
    cmd = "bedmap --sweep-all --faster --delim '\t' --multidelim '\t' --echo --echo-map  <(gunzip -c {muts}|grep -v \"^chrom\"|awk 'BEGIN{{FS=OFS=\"\t\"}} {{$2 = $2 OFS $2+1}} 1') <(zcat {annot})".format(annot=args.annotation, muts=args.input)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/bash')
    prev_chrom = prev_pos = None
    seen_chroms = set()
    n = 0
    for s in p.stdout:
        v = s.strip().split('\t')
        # v is mutation bed + extra columns when the mutation overlaps with a transcript
        # extra columns: chrom,start,end,dirs where dirs is either 1) +, 2) -, 3) +,-
        n_pos = n_neg = 0 
        if len(v) == n_cols + 1:
            pass  # no overlap
        else:
            strands = v[n_cols + 1 + 3]  # +1 for extra bed END column, +3 to get the strand from [chrom, start, end, strand]
            if strands not in ['+', '-', '+;-']:
                raise Exception('Unknown strand directionality {} at \n{}'.format(strands, s))
            if strands == '+':
                n_pos = 1
            elif strands == '-':
                n_neg = 1
            else:
                n_pos = n_neg = 1

#        n_pos = len(filter(lambda x: x == '+', strands))
#        n_neg = len(filter(lambda x: x == '-', strands))
#        st = None
        chrom, pos = v[0], int(v[1])
        if chrom != prev_chrom:
            if chrom in seen_chroms:
                sys.stderr.write('Input is not sorted (chromosome order): {}:{}\n'.format(chrom, pos))
                sys.exit(1)
            seen_chroms.add(chrom)
            prev_chrom = chrom
        else:
            if pos < prev_pos:
                sys.stderr.write('Input is not sorted (position order): {}:{}\n'.format(chrom, pos))
                sys.exit(1)
        prev_pos = pos
        ref, alt = v[3], v[4]
        base = reference[chrom][pos]
        if n_pos > 0:
            if n_neg > 0:
                st = '?'
            else:
                if base in ['C', 'T']:
                    st = '+'
                else:
                    st = '-'
        else:
            if n_neg > 0:
                if base in ['C', 'T']:
                    st = '-'
                else:
                    st = '+'
            else:
                st = '='
        o.write('{}\t{}\t{}\t{}\n'.format(chrom, pos, '\t'.join(v[3:n_cols + 1]), st))
        n += 1
        if (n % 1000000) == 0:
            sys.stdout.write('{}: {} mutations written\n'.format(datetime.datetime.now(), n))
    o.flush()
    o.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--annotation', default='/g/korbel/pitkaene/projects/deep_learning/mutation_autoencoder/data/Homo_sapiens.GRCh37.87.transcript_directionality.bed.gz')
    p.add_argument('--ref', default='/scratch/pitkaene/tmp/hs37d5_1000GP.fa')
    main(p.parse_args())
