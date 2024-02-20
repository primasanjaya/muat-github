#!/usr/bin/env python2

import sys, argparse, gzip, subprocess, random, datetime, itertools
from collections import deque
from preprocessing.dmm.util import read_reference, openz
import pdb
import numpy as np
import os
import pandas as pd
import shutil
import traceback
from natsort import natsort_keygen
import glob

dna_comp = {'A' : 'T', 'C' : 'G', 'G' : 'C', 'T' : 'A',
            'N' : 'N', '-' : '-', '+' : '+'}

accepted_pos = ['chr1',
'chr2',
'chr3',
'chr4',
'chr5',
'chr6',
'chr7',
'chr8',
'chr9',
'chr10',
'chr11',
'chr12',
'chr13',
'chr14',
'chr15',
'chr16',
'chr17',
'chr18',
'chr19',
'chr20',
'chr21',
'chr22',
'chrX',
'chrY']

accepted_pos_h19 = ['1',
'2',
'3',
'4',
'5',
'6',
'7',
'8',
'9',
'10',
'11',
'12',
'13',
'14',
'15',
'16',
'17',
'18',
'19',
'20',
'21',
'22',
'X',
'Y']

try:
    import swalign
    align_scoring = swalign.NucleotideScoringMatrix(2, -1)
    aligner = swalign.LocalAlignment(align_scoring, globalalign=True)
    support_complex = True
except:
    sys.stderr.write('Warning: module swalign not installed: complex variants ignored\n')
    sys.stderr.write('To install swalign: pip install swalign\n')
    support_complex = False

def dna_comp_default(x):
    r = dna_comp.get(x)
    return r if r is not None else x

def ispowerof2(x):
    return x != 0 and (x & (x - 1)) == 0

def get_context(v, prev_buf, next_buf, ref_genome,
                mutation_code, reverse_code, args):
    """Retrieve sequence context around the focal variant v, incorporate surrounding variants into
    the sequence."""
#    chrom, pos, fref, falt, vtype, _ = mut  # discard sample_id
    assert(args.context & (args.context - 1) == 0)
    flank = (args.context * 2) // 2 - 1
#    print('get_context', chrom, pos, fref, falt, args.context)
    if v.pos - flank - 1 < 0 or \
        (not args.no_ref_preload and v.pos + flank >= len(ref_genome[v.chrom])):
        return None
    if args.no_ref_preload:
        seq = subprocess.check_output(['samtools', 'faidx', args.reference,
                                      '{}:{}-{}'.format(v.chrom, v.pos - flank,
                                       v.pos + flank)])
        seq = ''.join(seq.decode().split('\n')[1:])
    else:
        seq = ref_genome[v.chrom][v.pos - flank - 1:v.pos + flank]
#    print('seqlen', len(seq))
    seq = list(seq)
    fpos = len(seq) // 2  # position of the focal mutation
    #for c2, p2, r2, a2, vt2, _ in itertools.chain(prev_buf, next_buf):
    for v2 in itertools.chain(prev_buf, next_buf):
        if args.nope:
            if v.pos != v2.pos or (v.pos == v2.pos and (v.ref != v2.ref or v.alt != v2.alt)):
                continue
        # make sure that the mutation stays in the middle of the sequence!
        assert(v2.chrom == v.chrom)
        tp = v2.pos - v.pos + flank
        if tp < 0 or tp >= len(seq):
            continue
#        print(c2, p2, r2, a2, vt2, len(r2), len(a2), len(seq))
        if v2.vtype == Variant.SNV:
            seq[tp] = mutation_code[v2.ref][v2.alt]
        elif v2.vtype == Variant.DEL:
            for i, dc in enumerate(v2.ref):
#                    print('DEL', i, dc, mutation_code[r2[i + 1]]['-'])
                seq[tp] = mutation_code[dc]['-']
                tp += 1
                if tp == len(seq):
                    break
            if v.pos == v2.pos:
#                    print('ADJ, del', fpos, (len(r2) - 1) / 2)
                fpos += len(v2.ref) // 2  # adjust to the deletion midpoint
        elif v2.vtype == Variant.INS:
            seq[tp] = seq[tp] + ''.join([mutation_code['-'][ic] for ic in v2.alt])
            if v2.pos < v.pos:      # adjust to the insertion midpoint
                # v2 is before focal variant - increment position by insertion length
                fpos += len(v2.alt)
            elif v2.pos == v.pos:
                # v2 is the focal variant - increment position by half of insertion length
                fpos += int(np.ceil(1.0 * len(v2.alt) / 2))
        elif v2.vtype == Variant.COMPLEX:
            if support_complex:
                m = align(v2.ref, v2.alt, mutation_code)  # determine mutation sequence
                if len(m) + tp >= len(seq):  # too long allele to fit into the context; increase context length
                    return None
                for i in range(len(m)):  # insert mutation sequence into original
                    seq[tp] = m[i]
                    tp += 1
                n_bp_diff = len(v2.alt) - len(v2.ref)
                if n_bp_diff > 0: # inserted bases? add to the end of the block, insertions are unrolled below
                    seq[tp - 1] = seq[tp - 1] + ''.join(v2.alt[len(v2.ref):])
                # we need to adjust the midpoint according to whether block is before or at the current midpoint
                if v2.pos < v.pos:
                    fpos += max(0, n_bp_diff)
                elif v2.pos == v.pos:
                    fpos += (len(m) - 1) // 2
        elif v2.vtype in Variant.MEI_TYPES:
            seq[tp] = seq[tp] + mutation_code['-'][v2.vtype]
            if v2.pos <= v.pos:  # handle SV breakpoints as insertions
                fpos += 1
        elif v2.vtype in Variant.SV_TYPES:
            seq[tp] = seq[tp] + mutation_code['-'][v2.vtype]
            if v2.pos <= v.pos:  # handle SV breakpoints as insertions
                fpos += 1
        elif v2.vtype == Variant.NOM:
            pass  # no mutation, do nothing (a negative datapoint)
        else:
            raise Exception('Unknown variant type: {}'.format(v2))
    # unroll any insertions and deletions (list of lists -> list)
    seq = [x for sl in list(map(lambda x: list(x), seq)) for x in sl]
    #print('seq2', seq)
    n = len(seq)
    # reverse complement the sequence if the reference base of the substitution is not C or T,
    # or the first inserted/deleted base is not C or T.
    # we transform both nucleotides and mutations here
#    print('UNRL fpos={}, seq={}, f="{}", seqlen={}'.format(fpos, ''.join(seq), seq[fpos], len(seq)))
    lfref, lfalt = len(v.ref), len(v.alt)
    if (lfref == 1 and lfalt == 1 and v.ref in 'AG') or \
       ((v.alt not in Variant.SV_TYPES) and (v.alt not in Variant.MEI_TYPES) and \
            ((lfref > 1 and v.ref[1] in 'AG') or (lfalt > 1 and v.alt[1]))):
        # dna_comp_default returns the input character for non-DNA characters (SV breakpoints)
        seq = map(lambda x: mutation_code[dna_comp_default(reverse_code.get(x)[0])]\
            [dna_comp_default(reverse_code.get(x)[1])], seq)[::-1]
        fpos = n - fpos - 1
#        print('REVC', fref, falt, 'fpos={}, seq={}, f="{}", seqlen={}'.format(fpos, ''.join(seq), seq[fpos], len(seq)))
    target_len = 2**int(np.floor(np.log2(args.context)))
    # trim sequence to length 2^n for max possible n
    #target_len = 2**int(np.floor(np.log2(n)))
    #trim = (n - target_len) / 2.0
    seq = ''.join(seq[max(0, fpos - int(np.floor(target_len / 2))):min(n, fpos + int(np.ceil(target_len / 2)))])
#    print('TRIM seqlen={}, tgtlen={}, seq={}, mid="{}"'.format(len(seq), target_len, ''.join(seq), seq[len(seq) // 2]))
    if len(seq) != target_len:
        return None
    return seq[3:6]

def is_valid_dna(s):
    s2 = [a in 'ACGTN' for a in s]
    return len(s2) == sum(s2)

def process_input(vr, o, sample_name, ref_genome, context,
                mutation_code, reverse_code, args):
    """A sweepline algorithm to insert mutations into the sequence flanking the focal mutation."""
    global warned_invalid_chrom
    prev_buf, next_buf = [], []
    i = 0
    n_var = n_flt = n_invalid = n_invalid_chrom = n_ok = 0
    for variant in vr:
        n_var += 1
        if args.report_interval > 0 and (n_var % args.report_interval) == 0:
            status('{} variants processed'.format(n_var), args)
        if args.no_ref_preload == False and variant.chrom not in ref_genome and variant.chrom != VariantReader.EOF:
            if warned_invalid_chrom == False:
                sys.stderr.write('Warning: a chromosome found in data not present in reference: {}\n'.format(variant.chrom))
                warned_invalid_chrom = True
            n_invalid_chrom += 1
            continue

        while len(next_buf) > 0 and (next_buf[0].chrom != variant.chrom or next_buf[0].pos < variant.pos - context):
            while len(prev_buf) > 0 and prev_buf[0].pos < next_buf[0].pos - context:
                prev_buf.pop(0)
            ctx = get_context(next_buf[0], prev_buf, next_buf, ref_genome,
                            mutation_code, reverse_code, args)
            if ctx is not None:
                # mutation not in the end of the chromosome and has full-len context
                next_buf[0].seq = ctx
                o.write(str(next_buf[0]) + '\n')
                n_ok += 1
            else:
                n_invalid += 1
            prev_buf.append(next_buf.pop(0))
        if len(prev_buf) > 0 and prev_buf[0].chrom != variant.chrom:
            prev_buf = []
        if variant.sample_id is None:
            variant.sample_id = sample_name   # id specific on per-file basis
        next_buf.append(variant)
    n_var -= 1  # remove terminator
    if args.verbose:
        n_all = vr.get_n_accepted() + vr.get_n_filtered()
        sys.stderr.write('{}/{} processed variants, {} filtered, {} invalid, {} missing chromosome\n'.\
            format(n_ok, n_all, vr.get_n_filtered(), n_invalid, n_invalid_chrom))
        sys.stderr.flush()

class Variant(object):
    SNV = 'SNV'        # single-nucleotide variant
    DEL = 'DEL'        # (small) deletion
    INS = 'INS'        # (small) insertion
    COMPLEX = 'CX'     # complex
    SV_DEL = 'SV_DEL'  # deletion
    SV_DUP = 'SV_DUP'  # duplication
    SV_INV = 'SV_INV'  # inversion
    SV_BND = 'SV_BND'  # breakend
    MEI_L1 = 'MEI_L1'  # LINE1 insertion
    MEI_ALU = 'MEI_ALU'# ALU insertion
    MEI_SVA = 'MEI_SVA'# SVA insertion
    MEI_PG = 'MEI_PG'  # Pseudogene insertion (?)
    UNKNOWN = 'UNKNOWN'
    NOM = 'NOM'        # NO Mutation
    SNV_TYPES = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    INDEL_TYPES = ['DEL', 'INS']
    SV_TYPES = [SV_DEL, SV_DUP, SV_INV, SV_BND]
    MEI_TYPES = [MEI_L1, MEI_ALU, MEI_SVA, MEI_PG]
    ALL_TYPES = [SNV, COMPLEX, UNKNOWN] + INDEL_TYPES + SV_TYPES + MEI_TYPES
    ALL_TYPES_SNV = SNV_TYPES + INDEL_TYPES + SV_TYPES + MEI_TYPES + [COMPLEX, UNKNOWN]
    SVCLASSES = {'DEL' : SV_DEL, 'DUP' : SV_DUP, 'INV' : SV_INV, 'TRA' : SV_BND}
    MEICLASSES = {'L1': MEI_L1, 'Alu': MEI_ALU, 'SVA': MEI_SVA, 'PG': MEI_PG }
    def __init__(self, chrom, pos, ref, alt, vtype=None, sample_id=None,
                 seq=None, extras=None):
        '''`alt` is either nucleotide sequence for SNVs and indels, or one of SV/MEI_TYPES.'''
        self.chrom = chrom
        self.pos = int(pos)
        self.ref = ref
        self.alt = alt
        self.seq = seq
        self.extras = extras
        if vtype is None:
            self.vtype = Variant.variant_type(ref, alt)
        else:
            self.vtype = vtype
        self.sample_id = sample_id
    def __str__(self):
        s = [self.chrom, self.pos, self.ref, self.alt, self.sample_id, self.seq]
        if self.extras is not None:
            s.extend(list(map(str, self.extras)))
        return '\t'.join(map(str, s))
    @staticmethod
    def variant_type(ref, alt, type_snvs=True):
        if len(ref) == 1 and len(alt) == 1:
            if type_snvs:
                if ref in 'AG':
                    ref, alt = dna_comp[ref], dna_comp[alt]
                return '{}>{}'.format(ref, alt)
            else:
                return Variant.SNV
        elif alt in Variant.SV_TYPES:
            return alt
        elif alt in Variant.MEI_TYPES:
            return alt
        elif len(alt) == 0:
            return Variant.DEL
        elif len(ref) == 0:
            return Variant.INS
        else:
            return Variant.COMPLEX
    @staticmethod
    def parse(line):
        '''Parse mutation data generated by process_input()'''
        v = line.strip().split('\t')
        chrom, pos, ref, alt, sample, seq = v[:6]
        extras = v[6:] if len(v) > 6 else None
        return Variant(chrom, pos, ref, alt, None, sample, seq, extras=extras)

class VariantReader(object):
    EOF = 'EOF'  # terminator token to signal end of input
    def __init__(self, f, pass_only=True, type_snvs=False, *args, **kwargs):
        self.f = f
        self.pass_only = pass_only
        self.type_snvs = type_snvs
        self.n_acc = self.n_flt = self.n_ref_alt_equal = 0
        self.eof = False
        self.prev_chrom = self.prev_pos = self.prev_sample = None
        self.seen_samples = set()
        self.seen_chroms = set()
    def get_n_accepted(self):
        return self.n_acc
    def get_n_filtered(self):
        return self.n_flt
    def update_pos(self, sample, chrom, pos):
        #assert(sample is not None and sample != "")
        if sample == self.prev_sample and chrom == self.prev_chrom and pos < self.prev_pos:
            sys.stderr.write('Error: input not sorted by position: {}:{}:{}\n'.format(sample, chrom, pos))
            sys.exit(1)
        if self.prev_sample != sample:
            if sample in self.seen_samples:
                sys.stderr.write('Error: sample already seen in input before: {}:{}:{}\n'.format(sample, chrom, pos))
                sys.exit(1)
            self.seen_samples.add(sample)
            self.seen_chroms = set()
        if self.prev_chrom != chrom:
            if chrom in self.seen_chroms:
                sys.stderr.write('Error: chromosomes not sorted: {}:{}:{}\n'.format(sample, chrom, pos))
                sys.exit(1)
            self.seen_chroms.add(chrom)
        self.prev_sample = sample
        self.prev_chrom = chrom
        self.prev_pos = pos

    @staticmethod
    def format(variant):
        raise NotImplementedError()
    @staticmethod
    def get_file_suffix():
        raise NotImplementedError()
    @staticmethod
    def get_file_sort_cmd(infn, hdrfn, outfn):
        raise NotImplementedError()
    def get_file_header(self):
        raise NotImplementedError()
    def __iter__(self):
        return self

class VCFReader(VariantReader):
    FILTER_PASS = ['.', 'PASS']
    SVCLASS_TO_SVTYPE = {'DEL' : Variant.SV_DEL, 'DUP' : Variant.SV_DUP,
                         't2tINV' : Variant.SV_INV, 't2hINV' : Variant.SV_INV,
                         'h2hINV' : Variant.SV_INV, 'h2tINV' : Variant.SV_INV,
                         'INV' : Variant.SV_INV, 'TRA' : Variant.SV_BND,
                         'L1': Variant.MEI_L1, 'Alu': Variant.MEI_ALU,
                         'SVA': Variant.MEI_SVA, 'PG': Variant.MEI_PG}
    SVTYPE_TO_SVCLASS = {Variant.SV_DEL : 'DEL', Variant.SV_DUP : 'DUP',
                         Variant.SV_INV : 'INV', Variant.SV_BND : 'TRA',
                         Variant.MEI_L1 : 'L1', Variant.MEI_ALU : 'Alu',
                         Variant.MEI_SVA: 'SVA'}


    def __init__(self, *args, **kwargs):
        super(VCFReader, self).__init__(*args, **kwargs)
        self.hdr = None
    def __next__(self):
        while 1:
            if self.eof:
                raise StopIteration()
            v = self.f.readline()
            if v.startswith('#'):
                if v.startswith('#CHROM'):
                    self.hdr = v
                continue
            if v == '':
                self.eof = True
                return Variant(chrom=VariantReader.EOF, pos=0, ref='N', alt='N')
            v = v.rstrip('\n').split('\t')
            chrom, pos, ref, alt, flt, info = v[0], int(v[1]), v[3], v[4], v[6], v[7]

            self.update_pos(None, chrom, pos)
            if self.pass_only and flt not in VCFReader.FILTER_PASS:
                self.n_flt += 1
                continue
            if ref == alt and ref != '':
                ref = alt = ''  # "T>T" -> "(null)>(null)"

            if alt[0] in '[]' or alt[-1] in '[]':  # SV, e.g., ]18:27105494]T
                info = dict([a for a in [c.split('=') for c in info.split(';')] if len(a) == 2])
                svc = info.get('SVCLASS', None)
                if svc is None:
                    sys.stderr.write('Warning: missing SVCLASS: {}:{}\n'.format(chrom, pos))
                    continue
                alt = VCFReader.SVCLASS_TO_SVTYPE.get(svc, None)
                if alt is None:
                    sys.stderr.write('Warning: unknown SVCLASS: {}:{}:{}\n'.format(chrom, pos, svc))
                    continue
            else:
                if is_valid_dna(ref) == False or is_valid_dna(alt) == False:
                    sys.stderr.write('Warning: invalid nucleotide sequence: {}:{}: {}>{}\n'.format(chrom, pos, ref, alt))
                    continue
                # canonize indels by removing anchor bases
                if len(ref) == 1 and len(alt) > 1:   # insertion
                    ref, alt = '', alt[1:]
                elif len(ref) > 1 and len(alt) == 1: # deletion
                    ref, alt = ref[1:], ''
                    pos += 1

            self.n_acc += 1
            # return None as sample id; vcf filename provides the sample id instead
            return Variant(chrom=chrom, pos=pos, ref=ref, alt=alt,
                           vtype=Variant.variant_type(ref, alt, self.type_snvs))

    def get_file_header(self):
        return '{}'.format(self.hdr.rstrip('\n'))

    @staticmethod
    def format(variant):
        "Convert the input variant into a string accepted by this reader"
        if variant.vtype in Variant.SV_TYPES or variant.vtype in Variant.MEI_TYPES:
            info = 'SVCLASS={}'.format(SVTYPE_TO_SVCLASS[variant.vtype])
        else:
            info = ''
        return '{}\t{}\t.\t{}\t{}\t.\t.\t{}\n'.format(variant.chrom, variant.pos, variant.ref, variant.alt, info)

    @staticmethod
    def get_file_suffix():
        return 'vcf'

    @staticmethod
    def get_file_sort_cmd(infn, hdrfn, outfn):
        return "/bin/bash -c \"cat {} <(LC_ALL=C sort -t $'\\t' -k1,1 -k2n,2 {}) >{}\"".format(hdrfn, infn, outfn)

class MAFReader(VariantReader):
    col_chrom_names = ['chromosome', 'chrom', 'chr']
    col_pos_names = ['position', 'chromosome_start', 'pos']
    col_ref_names = ['mutated_from_allele', 'ref', 'reference_genome_allele']
    col_alt_names = ['mutated_to_allele', 'alt']
    col_filter_names = ['filter']
    col_sample_names = ['sample']

    @staticmethod
    def find_col_ix(names, col_to_ix, fail_on_error=True):
        for n in names:
            if n in col_to_ix:
                return col_to_ix[n]
            if n.capitalize() in col_to_ix:
                return col_to_ix[n.capitalize()]
            if n.upper() in col_to_ix:
                return col_to_ix[n.upper()]
        if fail_on_error:
            sys.stderr.write('Could not find column(s) "{}" in header\n'.format(','.join(names)))
            sys.exit(1)
        else:
            return None

    def __init__(self, extra_columns=None, fake_header=False, *args, **kwargs):
        super(MAFReader, self).__init__(*args, **kwargs)
        if fake_header:
            self.hdr = MAFReader.create_fake_header(kwargs)
        else:
            self.hdr = self.f.readline().rstrip('\n')
        col_to_ix = dict([(x, i) for i, x in enumerate(self.hdr.split('\t'))])
        self.col_chrom_ix = MAFReader.find_col_ix(MAFReader.col_chrom_names, col_to_ix)
        self.col_pos_ix = MAFReader.find_col_ix(MAFReader.col_pos_names, col_to_ix)
        self.col_ref_ix = MAFReader.find_col_ix(MAFReader.col_ref_names, col_to_ix)
        self.col_alt_ix = MAFReader.find_col_ix(MAFReader.col_alt_names, col_to_ix)
        self.col_filter_ix = MAFReader.find_col_ix(MAFReader.col_filter_names, col_to_ix, fail_on_error=False)
        if kwargs['args'].sample_id:
            self.col_sample_ix = MAFReader.find_col_ix([kwargs['args'].sample_id], col_to_ix)
        else:
            self.col_sample_ix = MAFReader.find_col_ix(MAFReader.col_sample_names, col_to_ix)

        if extra_columns:
            self.extra_columns = list(map(col_to_ix.get, extra_columns))
            if None in self.extra_columns:
                raise Exception('Extra column(s) {} not found in input header\n{}'.format(\
                    extra_columns, self.extra_columns))
        else:
            self.extra_columns = []

    def __next__(self):
        while 1:
            if self.eof:
                raise StopIteration()
            v = self.f.readline()
            if v == '':
                self.eof = True
                return Variant(chrom=VariantReader.EOF, pos=0, ref='N', alt='N')
            v = v.rstrip('\n').split('\t')
            try:
                chrom, pos, ref, alt, sample = v[self.col_chrom_ix], \
                    int(v[self.col_pos_ix]), v[self.col_ref_ix], v[self.col_alt_ix], \
                    v[self.col_sample_ix]
                assert(sample != "")
                if self.col_filter_ix is None:
                    flt = 'PASS'
                else:
                    flt = v[self.col_filter_ix]
            except:
                print(v)
                raise
            self.update_pos(sample, chrom, pos)
            if self.pass_only and flt != 'PASS':
                self.n_flt += 1
                continue
            if ref == alt and ref != '':
                ref = alt = ''  # "T>T" -> "(null)>(null)"
            if (alt not in Variant.SV_TYPES) and (alt not in Variant.MEI_TYPES):
                # canonize indels
                if ref == '-' and len(alt) > 0:
                    ref = ''
                elif len(ref) > 0 and alt == '-':
                    alt = ''
            self.n_acc += 1
            extras = [v[ix] for ix in self.extra_columns]
            return Variant(chrom=chrom, pos=pos, ref=ref, alt=alt,
                           vtype=Variant.variant_type(ref, alt, type_snvs=self.type_snvs),
                           sample_id=sample, extras=extras)

    def format(self, variant):
        "Convert the input variant into a string accepted by this reader"
        v = ['.' for _ in range(len(self.hdr.split('\t')))]
        v[self.col_chrom_ix], v[self.col_pos_ix], \
            v[self.col_ref_ix], v[self.col_alt_ix], \
            v[self.col_filter_ix], v[self.col_sample_ix] = \
            variant.chrom, variant.pos, variant.ref, variant.alt, \
            'PASS', variant.sample_id
        for i, ix in enumerate(self.extra_columns):
            v[ix] = variant.info[i]
        return '{}\n'.format('\t'.join(map(str, v)))

    @staticmethod
    def create_fake_header(kwargs):
        if kwargs['args'].sample_id:
            sample_id = kwargs['args'].sample_id
        else:
            sample_id = MAFReader.col_sample_names[0]
        hdr = [MAFReader.col_chrom_names[0], MAFReader.col_pos_names[0],
               MAFReader.col_ref_names[0], MAFReader.col_alt_names[0],
               sample_id, MAFReader.col_filter_names[0]]
        return '\t'.join(hdr)

    @staticmethod
    def get_file_suffix():
        return 'maf'

    def get_file_header(self):
        return self.hdr

    def get_file_sort_cmd(self, infn, hdrfn, outfn, header=False):
        'Sort by sample, and then by chromosome and position'
        if header:
            return "/bin/bash -c \"cat {hdrfn} <(tail -n +2|LC_ALL=C sort -t $'\\t' -k{sample_ix},{sample_ix} -k{chrom_ix},{chrom_ix} -k{pos_ix}g,{pos_ix} {infn}) >{outfn}\"".format(\
                hdrfn=hdrfn, 
                sample_ix=self.col_sample_ix + 1, chrom_ix=self.col_chrom_ix + 1, pos_ix=self.col_pos_ix + 1,
                infn=infn, outfn=outfn)
        else:
            return "/bin/bash -c \"cat {hdrfn} <(LC_ALL=C sort -t $'\\t' -k{sample_ix},{sample_ix} -k{chrom_ix},{chrom_ix} -k{pos_ix}g,{pos_ix} {infn}) >{outfn}\"".format(\
                hdrfn=hdrfn, 
                sample_ix=self.col_sample_ix + 1, chrom_ix=self.col_chrom_ix + 1, pos_ix=self.col_pos_ix + 1,
                infn=infn, outfn=outfn)

def get_reader(f, args, type_snvs=False):
    if '.maf' in f.name:
        vr = MAFReader(f=f, pass_only=(args.no_filter == False), type_snvs=type_snvs,
                       extra_columns=args.info_column, args=args)
    elif '.vcf' in f.name:
        vr = VCFReader(f=f, pass_only=(args.no_filter == False), type_snvs=type_snvs)
    else:
        raise Exception('Unsupported file type: {}\n'.format(f.name))
    return vr


def open_stream(args,fn):
    if args.gel:
        sample_name = fn[:-7]
        sample_name = sample_name.split('/')
        sample_name = sample_name[0] + '_'.join(sample_name[1:])

        with gzip.open(fn, 'rb') as f_in:
            with open(args.tmp_dir + sample_name + '.vcf', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            
            f = open(args.tmp_dir + sample_name + '.vcf')
    else:
        if fn.endswith('.gz'):
            #f = gzip.open(fn)
            #sample_name = os.path.basename(fn).split('.')[0]
            sample_name = os.path.basename(fn).split('/')[0]
            sample_name = sample_name[:-7]

            with gzip.open(fn, 'rb') as f_in:
                with open(args.tmp_dir + sample_name + '.vcf', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            f = open(args.tmp_dir + sample_name + '.vcf')
        else:
            f = open(fn)
            sample_name = os.path.basename(fn).split('.')[0]
        assert(('.maf' in fn and '.vcf' in fn) == False)  # filenames should specify input type unambigiously
    return f, sample_name

def get_timestr():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def status(msg, args, lf=True, time=True):
    if args.verbose:
        if time:
            tstr = '[{}] '.format(get_timestr())
        else:
            tstr = ''
        sys.stderr.write('{}{}'.format(tstr, msg))
        if lf:
            sys.stderr.write('\n')
        sys.stderr.flush()

def read_codes(fn):
    codes = {}
    rcodes = {}
    for s in open(fn):
        ref, alt, code = s.strip().split()
        if ref not in codes:
            codes[ref] = {}
        codes[ref][alt] = code
        rcodes[code] = (ref, alt)
    rcodes['N'] = ('N', 'N')  # ->N, N>-, A>N etc all map to N, make sure that 'N'=>'N>N'
    return codes, rcodes

def func_annotate_mutation_all_modified(args):

    mutation_code, reverse_mutation_code = read_codes(args.mutation_coding)
    #if args.errors != '-':
    #    args.errf = open(args.errors, 'w')
    #else:
    #    args.errf = open(os.devnull, 'w')

    global warned_invalid_chrom
    warned_invalid_chrom = False

    "fns is all input file full paths in list"

    fns  = pd.read_csv(args.predict_filepath,sep='\t',index_col=0,low_memory=False)['path'].to_list()

    '''
    for ddir_or_fnlist in args.input:
        if os.path.isdir(ddir_or_fnlist):
            fns.extend(list(filter(accept_suffix,
                              [os.path.join(ddir_or_fnlist, d) for d in os.listdir(ddir_or_fnlist)])))
        else:
            fns.append(ddir_or_fnlist)
    '''

    n_missing = 0

    for fn in fns:
        if os.path.exists(fn) == False:
            sys.stderr.write('Input file {} not found\n'.format(fn))
            n_missing += 1
    if n_missing > 0:
        sys.exit(1)
    #pdb.set_trace()
    status('{} input files found'.format(len(fns)), args)
    if len(fns) == 0:
        sys.exit(1)

    if args.convert_hg38_hg19:
        status('Reading reference h38... ', args, lf=False)
        reference_38 = read_reference(args.reference_h38, args.verbose)   

        status('Reading reference h19... ', args, lf=False)
        reference_19 = read_reference(args.reference_h19, args.verbose)

    else:    
        status('Reading reference h19... ', args, lf=False)
        reference_19 = read_reference(args.reference_h19, args.verbose)

    # process variant

    try:
        os.makedirs(os.path.dirname(args.tmp_dir))
    except:
        pass

    all_error_file = []
    all_succeed_file = []
    for i, fn in enumerate(fns):
        try:
            #check VCF version:
            f = gzip.open(fn, 'rt')
            variable = f.readline()

            if variable == '##fileformat=VCFv4.2\n':
                version = '4.2'
            elif variable == '##fileformat=VCFv4.1\n':
                version = '4.1'
            else:
                print('current MuAt version cant process the VCF file version')
            #try:

            if args.gel:
                sample_name = fn[:-7]
                sample_name = sample_name.split('/')
                sample_name = sample_name[0] + '_'.join(sample_name[1:])
            else:
                get_ext = fn[-4:]
                if get_ext == '.vcf':
                    sample_name = fn[:-4]
                else:
                    sample_name = fn[:-7]
                #pdb.set_trace()
                sample_name = sample_name.split('/')
                sample_name = sample_name[-1]

            process = []

            if version == '4.1':
                output_file = args.tmp_dir + sample_name + '.tsv.gz'
                o = gzip.open(output_file, 'wt')
        
                f, sample_name_2 = open_stream(args,fn)

                digits = int(np.ceil(np.log10(len(fns))))
                fmt = '{:' + str(digits) + 'd}/{:' + str(digits) + 'd} {}: '
                if args.info_column:
                    infotag = '\t{}'.format('\t'.join(map(str.lower, args.info_column)))
                else:
                    infotag = ''
                status('Writing mutation sequences...', args)

                o.write('chrom\tpos\tref\talt\tsample\tseq{}\n'.format(infotag))
                
                status(fmt.format(i + 1, len(fns), sample_name), args)
                vr = get_reader(f, args)
                if args.convert_hg38_hg19:
                    process_input(vr, o, sample_name_2, reference_38, args.context,mutation_code, reverse_mutation_code, args)
                else:
                    process_input(vr, o, sample_name_2, reference_19, args.context,mutation_code, reverse_mutation_code, args)
                f.close()
                o.close()
                status('Output written to {}'.format(output_file), args)
                #pdb.set_trace()
                #natural sort motif
                pd_motif = pd.read_csv(output_file,sep='\t',low_memory=False,compression='gzip') 
                mutation = len(pd_motif)
                pd_motif  = pd_motif.sort_values(by=['chrom','pos'],key=natsort_keygen())
                pd_motif.to_csv(output_file,sep='\t',index=False, compression="gzip")
                #end motif
                process.append('motif')

            elif version == '4.2':
                import vcf
                vcf_reader = vcf.Reader(open(fn, 'rb'))
                output_file = args.tmp_dir + sample_name + '.tsv.gz'
                o = gzip.open(output_file, 'wt')
                o.write('chrom\tpos\tref\talt\tsample\tseq\n')

                mutation = 0
                for record in vcf_reader:
                    mutation = mutation + 1
                    if len(record.ALT) > 1:
                        pass
                    else:
                        if len(record.ALT[0])> 1:
                            #print('Warning: This version only process SNV, skip this mutation')
                            pass
                        else:
                            refbase = record.REF
                            ref_base_in_reference = reference_38[record.CHROM][record.POS-1]
                            if refbase != ref_base_in_reference:
                                #print('Warning: VCF file is not same as genome reference, please check the correct genome reference for this file')
                                pass
                            else:
                                #proceed mutation here 
                                first = reference_38[record.CHROM][record.POS-2]
                                mid_ref = reference_38[record.CHROM][record.POS-1]
                                #pdb.set_trace()
                                mid_sym = mutation_code[record.REF][str(record.ALT[0])]
                                third_ref = reference_38[record.CHROM][record.POS]

                                raw_seq = first + mid_ref + third_ref
                                seq = first + mid_sym + third_ref

                                if mid_ref == 'A' or mid_ref == 'G':
                                    revseq = seq[::-1]
                                    revseq = list(revseq)

                                    revcomp=[]
                                    for x in revseq:
                                        #pdb.set_trace()
                                        rev = mutation_code[dna_comp_default(reverse_mutation_code.get(x)[0])][dna_comp_default(reverse_mutation_code.get(x)[1])]
                                        revcomp.append(rev)
                                    
                                    #pdb.set_trace()
                                    revcomp = ''.join(revcomp)
                                    seq = revcomp
                                o.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(record.CHROM,record.POS,record.REF,record.ALT[0],sample_name,seq))
                o.close()

                #natural sort motif
                pd_motif = pd.read_csv(output_file,sep='\t',low_memory=False)
                print('original mutation:' + str(mutation) + ', preprocessed mutation:' + str(len(pd_motif)))
                pd_motif = pd_motif.sort_values(by=['chrom', 'pos'],key=natsort_keygen())
                pd_motif.to_csv(output_file,sep='\t',index=False, compression="gzip")
                #end motif
                process.append('motif')

            else:
                print('todo other vcf version')
            
            if args.convert_hg38_hg19:
                #lift over comes here
                from pyliftover import LiftOver

                #lo = LiftOver('/mnt/g/experiment/redo_muat/muat-github/preprocessing/genomic_tracks/hg38ToHg19.over.chain.gz')
                #lo = LiftOver('/genomic_tracks/GRCh37_to_GRCh38.chain.gz')
                lo = LiftOver('hg38', 'hg19')
                
                pd_hg38 = pd.read_csv(output_file,sep='\t',low_memory=False) 
                chrom_pos = []


                for i in range(len(pd_hg38)):
                    try:
                        row = pd_hg38.iloc[i]
                        chrom = row['chrom']
                        pos = row['pos']
                        ref = row['ref']
                        alt = row['alt']
                        sample = row['sample']
                        seq = row['seq']
                        #gc1kb = row['gc1kb']
                        hg19chrompos = lo.convert_coordinate(chrom, pos)
                        chrom = hg19chrompos[0][0][3:]
                        pos = hg19chrompos[0][1]
                        chrom_pos.append((chrom,pos,ref,alt,sample,seq))
                    except:
                        print('cant be converted at pos ' +str(row['chrom']) +':'+ str(row['pos']))
                pd_hg19 = pd.DataFrame(chrom_pos)

                #pdb.set_trace()
                pd_hg19.columns = pd_hg38.columns.tolist()
                #natural sort
                pd_hg19 = pd_hg19.loc[pd_hg19['chrom'].isin(accepted_pos_h19)]
                pd_hg19 = pd_hg19.sort_values(by=['chrom','pos'], key=natsort_keygen())
                pd_hg19.to_csv(output_file,sep='\t',index=False, compression="gzip")
                #end liftover
                process.append('liftover')

                mut_liftover = len(pd_hg19)
            else:
                mut_liftover = 0
                process.append('no-liftover')

            #next gc content
            input_gc = output_file
            output_gc = args.tmp_dir + sample_name + '.gc.tsv.gz'
            label = 'gc1kb'
            verbose = True
            window = 1001
            batch_size = 100000
            
            #gc content
            '''
            syntax_gc = 'python3 preprocessing/dmm/annotate_mutations_with_gc_content.py \
            -i ' + args.tmp_dir + only_input_filename + '.tsv.gz \
            -o ' + args.tmp_dir + only_input_filename + '.gc.tsv.gz \
            -n 1001 \
            -l gc1kb \
            --reference ' + args.reference + ' \
            --verbose'
            '''

            o = openz(output_gc, 'wt')

            #pdb.set_trace()
        
            hdr = openz(input_gc).readline().strip().decode("utf-8").split('\t')
            n_cols = len(hdr)
            sys.stderr.write('{} columns in input\n'.format(n_cols))
            if hdr[0] == 'chrom':
                sys.stderr.write('Header present\n')
                #pdb.set_trace()
                o.write('{}\t{}\n'.format('\t'.join(hdr), label))
            else:
                sys.stderr.write('Header absent\n')
                hdr = None

            n_mut = 0
            then = datetime.datetime.now()
            with openz(input_gc) as f:
                if hdr:
                    f.readline()
                cchrom = None
                for s in f:
                    v = s.strip().decode("utf-8").split('\t')
                    chrom, pos = v[0], int(v[1])
                    if cchrom != chrom:
                        cchrom = chrom
                        cpos = max(0, pos - window / 2)
                        mpos = min(len(reference_19[chrom]) - 1, cpos + window)
                        cpos = round(cpos)
                        mpos = round(mpos)
                        buf = deque(reference_19[chrom][cpos:mpos])
                        gc = sum([1 for c in buf if c == 'C' or c == 'G'])
                        at = sum([1 for c in buf if c == 'A' or c == 'T'])
                    else:
                        cpos2 = max(0, pos - window / 2)
                        cdiff = cpos2 - cpos
                        if cdiff > 0:
                            if cdiff < window:
                                # incremental update of buffer
                                for _ in range(round(cdiff)):
                                    remove = buf.popleft()
                                    if remove == 'C' or remove == 'G':
                                        gc -= 1
                                    elif remove == 'A' or remove == 'T':
                                        at -= 1
                                insert = reference_19[cchrom][round(cpos+window):round(cpos+window+cdiff)]
                                gc += sum([1 for c in insert if c == 'C' or c == 'G'])
                                at += sum([1 for c in insert if c == 'A' or c == 'T'])
                                buf.extend(insert)
                            else:
                                # reinit buffer at cpos2
                                mpos = min(len(reference_19[chrom]) - 1, cpos2 + window)
                                buf = deque(reference_19[chrom][round(cpos2):round(mpos)])
                                gc = sum([1 for c in buf if c == 'C' or c == 'G'])
                                at = sum([1 for c in buf if c == 'A' or c == 'T'])
                            cpos = cpos2
                    try:
                        gc_ratio = 1.0 * gc / (gc + at)
                    except:
                        gc_ratio = 0

                    o.write('{}\t{}\n'.format(s.strip().decode("utf-8"), gc_ratio))

                    n_mut += 1
                    if verbose and (n_mut % batch_size) == 0:
                        now = datetime.datetime.now()
                        td = now - then
                        sys.stderr.write('{}: {} mutations. {}/mutation\n'.format(now, n_mut, td / batch_size))
                        then = now
            o.close()
            
            #natural sort #remove nan gc
            #pdb.set_trace()
            pd_sort = pd.read_csv(output_gc,sep='\t',low_memory=False,compression="gzip")
            #remove nan genic
            pd_sort = pd_sort[~pd_sort['gc1kb'].isna()]
            pd_sort['chrom'] = pd_sort['chrom'].astype('string')
            pd_sort = pd_sort.loc[pd_sort['chrom'].isin(accepted_pos_h19)]
            pd_sort = pd_sort.sort_values(by=['chrom','pos'],key=natsort_keygen())
            pd_sort.to_csv(output_gc,sep='\t',index=False, compression="gzip")

            process.append('gc')

            # Genic region
            output_genic = args.tmp_dir + sample_name + '.gc.genic.tsv.gz'
            syntax_genic = 'preprocessing/dmm/annotate_mutations_with_bed.sh \
            ' + output_gc + ' \
            ' + args.genomic_tracks + 'Homo_sapiens.GRCh37.87.genic.genomic.bed.gz \
            '+ output_genic + '\
            genic'
            subprocess.run(syntax_genic, shell=True)

            #natural sort #remove nan genic
            #pdb.set_trace()
            pd_sort = pd.read_csv(output_genic,sep='\t',low_memory=False,compression="gzip")
            #remove nan genic
            pd_sort = pd_sort[~pd_sort['genic'].isna()]
            pd_sort['chrom'] = pd_sort['chrom'].astype('string')
            pd_sort = pd_sort.loc[pd_sort['chrom'].isin(accepted_pos_h19)]
            pd_sort = pd_sort.sort_values(by=['chrom', 'pos'],key=natsort_keygen())
            pd_sort.to_csv(output_genic,sep='\t',index=False, compression="gzip")
            process.append('genic')
            
            #exon regions
            output_exon = args.tmp_dir + sample_name + '.gc.genic.exonic.tsv.gz'

            syntax_exonic = 'preprocessing/dmm/annotate_mutations_with_bed.sh \
            ' + output_genic + ' \
            ' + args.genomic_tracks + 'Homo_sapiens.GRCh37.87.exons.genomic.bed.gz \
            ' + output_exon + ' \
            exonic'
            subprocess.run(syntax_exonic, shell=True)

            pd_sort = pd.read_csv(output_exon,sep='\t',low_memory=False,compression="gzip") 
            pd_sort = pd_sort[~pd_sort['exonic'].isna()]
            pd_sort['chrom'] = pd_sort['chrom'].astype('string')
            pd_sort = pd_sort.loc[pd_sort['chrom'].isin(accepted_pos_h19)]
            pd_sort = pd_sort.sort_values(by=['chrom', 'pos'],key=natsort_keygen())
            pd_sort.to_csv(output_exon,sep='\t',index=False, compression="gzip")

            process.append('exonic')

            output_cs = args.tmp_dir + sample_name + '.gc.genic.exonic.cs.tsv.gz'
            annotation = args.genomic_tracks + 'Homo_sapiens.GRCh37.87.transcript_directionality.bed.gz'

            o = openz(output_cs, 'wt')

            hdr = openz(output_exon).readline().strip().decode("utf-8").split('\t')
            n_cols = len(hdr)
            sys.stderr.write('{} columns in input\n'.format(n_cols))
            if hdr[0] == 'chrom':
                sys.stderr.write('Header present\n')
                o.write('{}\tstrand\n'.format('\t'.join(hdr)))
            else:
                sys.stderr.write('Header absent\n')
                hdr = None
            sys.stderr.write('Reading reference: ')
            #reference = read_reference(args.ref, verbose=True)
            cmd = "bedmap --sweep-all --faster --delim '\t' --multidelim '\t' --echo --echo-map  <(gunzip -c {muts}|grep -v \"^chrom\"|awk 'BEGIN{{FS=OFS=\"\t\"}} {{$2 = $2 OFS $2+1}} 1') <(zcat {annot})".format(annot=annotation, muts=output_exon)
            #pdb.set_trace()
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/bash')
            prev_chrom = prev_pos = None
            seen_chroms = set()
            n = 0
            for s in p.stdout:
                v = s.strip().decode("utf-8").split('\t')
                # v is mutation bed + extra columns when the mutation overlaps with a transcript
                # extra columns: chrom,start,end,dirs where dirs is either 1) +, 2) -, 3) +,-
                n_pos = n_neg = 0 
                if len(v) == n_cols + 1:
                    pass  # no overlap
                else:
                    try:
                        strands = v[n_cols + 1 + 3]  # +1 for extra bed END column, +3 to get the strand from [chrom, start, end, strand]
                        if strands not in ['+', '-', '+;-']:
                            raise Exception('Unknown strand directionality {} at \n{}'.format(strands, s))
                        if strands == '+':
                            n_pos = 1
                        elif strands == '-':
                            n_neg = 1
                        else:
                            n_pos = n_neg = 1
                    except:
                        pdb.set_trace()

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
                #pdb.set_trace()
                base = reference_19[chrom][pos]
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
            
            pd_sort = pd.read_csv(output_cs,sep='\t',low_memory=False,compression="gzip") 
            pd_sort = pd_sort[~pd_sort['strand'].isna()]

            pd_sort['chrom'] = pd_sort['chrom'].astype('string')
            pd_sort = pd_sort.loc[pd_sort['chrom'].isin(accepted_pos_h19)]
            pd_sort = pd_sort.sort_values(by=['chrom', 'pos'],key=natsort_keygen())
            preprocessed_mutation = len(pd_sort)
            pd_sort.to_csv(output_cs,sep='\t',index=False, compression="gzip")

            process.append('strand')

            if args.convert_hg38_hg19:
                if process == ['motif','liftover','gc','genic','exonic','strand']:
                    tup_mut = [(mutation,mut_liftover, preprocessed_mutation)]
                    pd_complete_mutation = pd.DataFrame(tup_mut)
                    pd_complete_mutation.columns = ['original_mutation','liftover_mutation','preprocessed_mutation']
                    pd_complete_mutation.to_csv(args.tmp_dir + sample_name + '_mutations.csv')

                    #task complete
                    #pdb.set_trace()
                    try:
                        os.remove(args.tmp_dir + sample_name + '.vcf')
                    except:
                        pass
                    os.remove(args.tmp_dir + sample_name + '.tsv.gz')
                    os.remove(args.tmp_dir + sample_name + '.gc.tsv.gz')
                    os.remove(args.tmp_dir + sample_name + '.gc.genic.tsv.gz')
                    os.remove(args.tmp_dir + sample_name + '.gc.genic.exonic.tsv.gz')
                    all_succeed_file.append(fn)
                else:
                    all_files = glob.glob(args.tmp_dir + sample_name + '*')
                    for i_files in range(len(all_files)):
                        os.remove(all_files[i_files])
            else:
                if process == ['motif','no-liftover','gc','genic','exonic','strand']:
                    tup_mut = [(mutation,mut_liftover,preprocessed_mutation)]
                    pd_complete_mutation = pd.DataFrame(tup_mut)
                    pd_complete_mutation.columns = ['original_mutation','liftover_mutation','preprocessed_mutation']
                    pd_complete_mutation.to_csv(args.tmp_dir + sample_name + '_mutations.csv')
                    #task complete

                    try:
                        os.remove(args.tmp_dir + sample_name + '.vcf')
                    except:
                        pass
                    os.remove(args.tmp_dir + sample_name + '.tsv.gz')
                    os.remove(args.tmp_dir + sample_name + '.gc.tsv.gz')
                    os.remove(args.tmp_dir + sample_name + '.gc.genic.tsv.gz')
                    os.remove(args.tmp_dir + sample_name + '.gc.genic.exonic.tsv.gz')
                    all_succeed_file.append(fn)
                else:
                    all_files = glob.glob(args.tmp_dir + sample_name + '*')
                    for i_files in range(len(all_files)):
                        os.remove(all_files[i_files])
        except Exception as err:

            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno),type(err).__name__, err)
            all_files = glob.glob(args.tmp_dir + sample_name + '*')
            for i_files in range(len(all_files)):
                os.remove(all_files[i_files])

    if len(all_error_file)>0:
        pd_all_error_file = pd.DataFrame(all_error_file)
        pd_all_error_file.columns = ['path']
        pd_all_error_file.to_csv(args.tmp_dir + 'error_file.tsv',sep='\t')

    if len(all_succeed_file)>0:
        pd_all_succeed_file = pd.DataFrame(all_succeed_file)
        pd_all_succeed_file.columns = ['path']
        pd_all_succeed_file.to_csv(args.tmp_dir + 'succeed_file.tsv',sep='\t')