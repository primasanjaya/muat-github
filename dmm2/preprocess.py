'''
Preprocess mutation data by adding to each mutation the sequence context from a reference genome
and neighboring mutations.

Input is read from VCF or MAF files.
For VCF files, REF and ALT for insertions and deletions are assumed
to contain the unmodified (anchoring) reference base and not '-', i.e. AATG>A instead of ATG>-
and G>GA instead of ->A. POS is expected to be the coordinate of this unmodified
reference base.
For MAF files, REF and ALT must not contain the anchoring base.

Complex variants (|ref|>1 and |alt|>1) are processed by first aligning
ref to alt, and then inserting the best alignment into the sequence.

SV breakpoints are supported by inserting a breakpoint character into the
sequence.

Negative examples can be added to the dataset with --generate-negatives.
This randomly picks mutation positions from input for each sample, and
inserts the positions into variant input without any base change.

Mutations and sequence context are encoded using codes specified
in a coding table (data/mutation_codes.tsv by default).
Mutation/dna string is reverse complemented if the reference base of a substitution,
or the first inserted/deleted base of an indel is either A or G.
Small deletions and insertions are positioned to have approximately the same amount of flanking sequence
on both the 5' and 3' sides.

Examples:
AG[C>A]TG -> AG$TG
TC[A>C]CA -> TG?GA (ref base A or G => reverse complement string)
AG[T>-]TT -> AG4TT
AG[A>-]TT -> AA1CT (reverse complement)
AG[AG>-]TT -> AA24CT (2 bp deletion, reverse complement)
CT[->T]AA -> CT8AA
AA[->T]CC -> GG1TT (5' base on the + strand is A or G: reverse complement)
CT[->TT]AA -> CT88AA
CT[SV_DEL]AA -> CTDAA

A simple sweepline algorithm to insert mutations into the flanking sequence:
----------------------------------------------------------------------------
For each mutation at position p:
    while next_buf[0].pos < p + window or next_buf[0].chrom != mutation.chrom:
        while last_buf[0].pos < next_buf[0].pos - window:
            last_buf.pop(0)
        output sequence centered at next_buf[0].pos (last_buf + next_buf)
        append next_buf[0].pop(0) to prev_buf
    append mutation to next_buf
'''

import sys, os, argparse, gzip, subprocess, itertools, datetime, tempfile
import json, shutil, math, random, re, collections
import numpy as np

from common import *

MODE_NEG_GENERATE = 'generate'
MODE_NEG_AUGMENT = 'augment'
MODE_NEG_PROCESS = 'process'

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
            s.extend(map(str, self.extras))
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

# translation table to map each character to a nucleotide or N
valid_dna = ''.join([chr(x) if chr(x) in 'ACGTN' else 'N' for x in range(256)])

def is_valid_dna(s):
    s2 = [a in 'ACGTN' for a in s]
    return len(s2) == sum(s2)

def read_reference(reffn, verbose=0):
    R = {}
    chrom = None
    if reffn.endswith('.gz'):
        f = gzip.open(reffn)
    else:
        f = open(reffn)
    for s in f:
        if s[0] == '>':
            if chrom is not None:
                R[chrom] = ''.join(seq).translate(valid_dna)
            seq = []
            chrom = s[1:].strip().split()[0]
            if verbose:
                sys.stderr.write('{} '.format(chrom))
                sys.stderr.flush()
        else:
            seq.append(s.strip().upper())
    R[chrom] = ''.join(seq).translate(valid_dna)
    if verbose:
        sys.stderr.write(' done.\n')
    return R

re_ext_cigar = re.compile('(\d+)([MXID])')

def align(ref, alt, mutation_code):
    alignment = aligner.align(ref, alt)
    ix_r = ix_a = 0
    s = []
    for seg_length, seg_type in re_ext_cigar.findall(alignment.extended_cigar_str):
        seg_length = int(seg_length)
        # seg_type is M, X, D or I
        if seg_type == 'M' or seg_type == 'X':
            s.extend([mutation_code[ref[ix_r + i]][alt[ix_a + i]] for i in range(seg_length)])
            ix_r += seg_length
            ix_a += seg_length
        elif seg_type == 'D':
            s.extend([mutation_code[ref[ix_r + i]]['-'] for i in range(seg_length)])
            ix_r += seg_length
        elif seg_type == 'I':
            s.extend([mutation_code['-'][alt[ix_a + i]] for i in range(seg_length)])
            ix_a += seg_length
        else:
            assert(0)  # invalid seg_type
    return s

def get_context(v, prev_buf, next_buf, ref_genome,
                mutation_code, reverse_code, args):
    """Retrieve sequence context around the focal variant v, incorporate surrounding variants into
    the sequence."""
#    chrom, pos, fref, falt, vtype, _ = mut  # discard sample_id
    assert(ispowerof2(args.context))
    flank = (args.context * 2) / 2 - 1
#    print 'get_context', chrom, pos, fref, falt, args.context
    if v.pos - flank - 1 < 0 or \
        (args.no_ref_preload == False and v.pos + flank >= len(ref_genome[v.chrom])):
        return None
    if args.no_ref_preload:
        seq = subprocess.check_output(['samtools', 'faidx', args.reference,
                                      '{}:{}-{}'.format(v.chrom, v.pos - flank,
                                       v.pos + flank)])
        seq = ''.join(seq.split('\n')[1:])
    else:
        seq = ref_genome[v.chrom][v.pos - flank - 1:v.pos + flank]
#    print 'seqlen', len(seq)
    seq = list(seq)
    fpos = len(seq) / 2  # position of the focal mutation
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
#        print c2, p2, r2, a2, vt2, len(r2), len(a2), len(seq)
        if v2.vtype == Variant.SNV:
            seq[tp] = mutation_code[v2.ref][v2.alt]
        elif v2.vtype == Variant.DEL:
            for i, dc in enumerate(v2.ref):
#                    print 'DEL', i, dc, mutation_code[r2[i + 1]]['-']
                seq[tp] = mutation_code[dc]['-']
                tp += 1
                if tp == len(seq):
                    break
            if v.pos == v2.pos:
#                    print 'ADJ, del', fpos, (len(r2) - 1) / 2
                fpos += len(v2.ref) / 2  # adjust to the deletion midpoint
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
                    fpos += (len(m) - 1) / 2
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
    #print 'seq2', seq
    n = len(seq)
    # reverse complement the sequence if the reference base of the substitution is not C or T,
    # or the first inserted/deleted base is not C or T.
    # we transform both nucleotides and mutations here
#    print 'UNRL fpos={}, seq={}, f="{}", seqlen={}'.format(fpos, ''.join(seq), seq[fpos], len(seq))
    lfref, lfalt = len(v.ref), len(v.alt)
    if (lfref == 1 and lfalt == 1 and v.ref in 'AG') or \
       ((v.alt not in Variant.SV_TYPES) and (v.alt not in Variant.MEI_TYPES) and \
            ((lfref > 1 and v.ref[1] in 'AG') or (lfalt > 1 and v.alt[1]))):
        # dna_comp_default returns the input character for non-DNA characters (SV breakpoints)
        seq = map(lambda x: mutation_code[dna_comp_default(reverse_code.get(x)[0])]\
            [dna_comp_default(reverse_code.get(x)[1])], seq)[::-1]
        fpos = n - fpos - 1
#        print 'REVC', fref, falt, 'fpos={}, seq={}, f="{}", seqlen={}'.format(fpos, ''.join(seq), seq[fpos], len(seq))
    target_len = 2**int(np.floor(np.log2(args.context)))
    # trim sequence to length 2^n for max possible n
    #target_len = 2**int(np.floor(np.log2(n)))
    #trim = (n - target_len) / 2.0
    seq = ''.join(seq[max(0, fpos - int(np.floor(target_len / 2))):min(n, fpos + int(np.ceil(target_len / 2)))])
#    print 'TRIM seqlen={}, tgtlen={}, seq={}, mid="{}"'.format(len(seq), target_len, ''.join(seq), seq[len(seq) / 2])
    if len(seq) != target_len:
        return None
    return seq

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
        assert(sample is not None and sample != "")
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
    def next(self):
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
                info = dict(filter(lambda a: len(a) == 2, [c.split('=') for c in info.split(';')]))
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
            self.extra_columns = map(col_to_ix.get, extra_columns)
            if None in self.extra_columns:
                raise Exception('Extra column(s) {} not found in input header\n{}'.format(\
                    extra_columns, self.extra_columns))
        else:
            self.extra_columns = []

    def next(self):
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
                print v
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

FILE_SUFFIXES = ['.vcf', '.vcf.gz', '.maf', '.maf.gz']

def strip_suffixes(s, suffixes):
    loop = True
    while loop:
        loop = False
        for suf in suffixes:
            if s.endswith(suf):
                s = s[:-len(suf)]
                loop = True
    return s

def accept_suffix(s, suffixes=FILE_SUFFIXES):
    for suf in suffixes:
        if s.endswith(suf):
            return True
    return False

def open_stream(fn):
    if fn.endswith('.gz'):
        f = gzip.open(fn)
        sample_name = os.path.basename(fn).split('.')[0]
    else:
        f = open(fn)
        sample_name = os.path.basename(fn).split('.')[0]
    assert(('.maf' in fn and '.vcf' in fn) == False)  # filenames should specify input type unambigiously
    return f, sample_name

def get_reader(f, args, type_snvs=False):
    if '.maf' in f.name:
        vr = MAFReader(f=f, pass_only=(args.no_filter == False), type_snvs=type_snvs,
                       extra_columns=args.info_column, args=args)
    elif '.vcf' in f.name:
        vr = VCFReader(f=f, pass_only=(args.no_filter == False), type_snvs=type_snvs)
    else:
        raise Exception('Unsupported file type: {}\n'.format(f.name))
    return vr

def prepare_negative_examples(fns, args):
    vtype_to_ix = {}
    for vtype in Variant.ALL_TYPES_SNV:
        vtype_to_ix[vtype] = []
    variant_to_neg_sample = {}  # variant index to neg sample index
    sample_vtype_counts = {}
    sample_to_ix = {}
    sample_ix_to_id = {}
    sample_to_variants = {}

    # store variant indexes
    vix = 0
    # this iteration must match the one in process_input for the variant indexes to match
    status('Negative generation: stratify variants by type', args)
    for i, fn in enumerate(fns):
        status('{}/{}: Stratify variants in {}'.format(i + 1, len(fns), fn), args)
        f, sample_name = open_stream(fn)
        for variant in get_reader(f, args, type_snvs=True):
            if variant.chrom == VariantReader.EOF:
                continue
            if variant.sample_id is None:
                sample_id = sample_name
            else:
                sample_id = variant.sample_id
            if sample_id not in sample_vtype_counts:
                sample_vtype_counts[sample_id] = collections.Counter()
                sample_to_variants[sample_id] = set()
            sample_vtype_counts[sample_id][variant.vtype] += 1
            vtype_to_ix[variant.vtype].append(vix)
            sample_to_variants[sample_id].add(vix)
            variant_to_neg_sample[vix] = []
            vix += 1
    status('{} variants counted in {} samples'.format(vix, len(sample_vtype_counts)), args)

    # count median variant count
    c = []
    for sample_id in sample_vtype_counts:
        cc = []
        for vt in Variant.ALL_TYPES_SNV:
            if vt in sample_vtype_counts[sample_id]:
                cc.append(sample_vtype_counts[sample_id][vt])
            else:
                cc.append(0)
        c.append(cc)
    median_vtype_counts = dict(zip(Variant.ALL_TYPES_SNV, np.median(np.array(c), axis=0)))
    median_fn = '{}.vtype_medians.json'.format(strip_suffixes(args.output, ['.tsv.gz']))
    with open(median_fn, 'w') as o:
        json.dump(median_vtype_counts, o)
        sys.stderr.write('Wrote variant type median counts to {}\n'.format(median_fn))

    # assign negative examples for each variant
    status('Negative generation: assign negatives to samples', args)
    next_sample_ix = 0
    sample_count = len(sample_vtype_counts)
    for i, sample_id in enumerate(sorted(sample_vtype_counts.keys())):
        status('{}/{}: Assign negatives to in {}'.format(i + 1, sample_count, sample_id), args)
        sample_to_ix[sample_id] = next_sample_ix
        sample_ix_to_id[next_sample_ix] = sample_id
        next_sample_ix += 1
        for vtype in Variant.ALL_TYPES_SNV:
            if args.median_variant_type_negatives:
                # for each variant type generate median number of variants
                # in the dataset multiplied by --generate-negatives
                n = median_vtype_counts[vtype]
            else:
                # for each variant type in each sample generate as many negative
                # examples multiplied by --generate-negatives
                n = sample_vtype_counts[sample_id][vtype]
            #sys.stderr.write('Negs {} {} {}\n'.format(sample_id, vtype, n))
            rn = int(math.ceil(1.0 * args.generate_negatives * n))
            if rn == 0:
                continue
            tries = rn * 10
            n_negs = 0
            while n_negs < rn and tries > 0:
                # negative variant must not be from the same sample
                ix = random.choice(vtype_to_ix[vtype])
                if ix not in sample_to_variants[sample_id]:
                    variant_to_neg_sample[ix].append(sample_to_ix[sample_id])
                    #sys.stderr.write('Assign neg {} {} {}\n'.format(sample_id, vtype, ix, sample_to_ix[sample_id]))
                    n_negs += 1
                else:
                    tries -= 1
            if n_negs < rn:
                status('Warning: unable to generate negatives for {}:{}:{}/{}'.format(sample_id, vtype, n_negs, rn), args)

    return variant_to_neg_sample, sample_ix_to_id

def generate_negatives(fns, variant_to_neg_sample, sample_ix_to_id, args):
    '''Read in variant input, augment with negative examples.
Output written as a MAF file to have per-variant sample ids.'''
    ofns = []
    i = 0
    odir1 = tempfile.mkdtemp(dir=args.tmp)
    odir2 = tempfile.mkdtemp(dir=args.tmp)
#    of = tempfile.NamedTemporaryFile(mode='w', delete=False)
    try:
        # 1st phase: process each input file, add negative examples, output a single augmented file
        maf = MAFReader(f=None, fake_header=True, args=args)
        hdrf = tempfile.NamedTemporaryFile()
        hdrf.write('{}\n'.format(maf.get_file_header()))
        hdrf.flush()
        with tempfile.NamedTemporaryFile(dir=odir1, delete=False) as of:
        #print 'Writing temp to {}...'.format(of.name)
        # this iteration must match the ones in prepare_negatives and process_input
        # for the variant indexes to match
            for j, fn in enumerate(fns):  # for each input file, create a temp file containing input augmented with negative examples
                status('{}/{}: Augmenting input {}...'.format(j + 1, len(fns), fn), args)
                f, sample_name = open_stream(fn)
                vr = get_reader(f, args, type_snvs=True)
                for variant in vr:
                    if variant.chrom == VariantReader.EOF:
                        continue
                    if args.report_interval > 0 and (i % args.report_interval) == 0:
                         status('{} variants processed (dataset augmentation)'.format(i), args)
                    if variant.sample_id is None:
                        variant.sample_id = sample_name
                    for r_sample_id in map(sample_ix_to_id.get, variant_to_neg_sample[i]):
                        assert(r_sample_id is not None)
                        of.write(maf.format(Variant(chrom=variant.chrom, pos=variant.pos,
                                                    ref='', alt='',
                                                    vtype=Variant.NOM,
                                                    sample_id=r_sample_id, extras=variant.extras)))
                    assert(variant.sample_id is not None)
                    of.write(maf.format(variant))
                    i += 1

        # 2nd phase: sort the augmented file by sample
        sortedfn = os.path.join(odir1, 'sorted.maf.gz')
        sortcmd = maf.get_file_sort_cmd(infn=of.name, hdrfn=hdrf.name,
                                        outfn=sortedfn, header=False)
        subprocess.call(sortcmd, shell=True)

        # 3rd phase: split the augmented file by sample
        status('Negative generation: stratify augmented variants by sample', args)
        split_fns = []
        with open(sortedfn) as f:
            f.readline() # skip header
            cur_sample_id = of = None
            for s in f:
                sample_id = s.strip().split('\t')[maf.col_sample_ix]
                if sample_id != cur_sample_id:
                    if of is not None:
                        of.close()
                    cur_sample_id = sample_id
                    fn = os.path.join(odir2, '{}.maf'.format(sample_id))
#                    split_fns.append((sample_id, fn))
                    ofns.append(fn)
                    of = open(fn, 'w')
                    of.write('{}\n'.format(maf.get_file_header()))
                of.write(s)
            if of is not None:
                of.close()

        # # 4th phase: sort each augmented sample file
        # for sample_id, sample_fn in split_fns:
        #     status('Sorting augmented input for {}...'.format(sample_id), args)
        #     sortedfn = os.path.join(odir2, '{}.maf'.format(sample_id))
        #     sortcmd = MAFReader.get_file_sort_cmd(infn=sample_fn,
        #                                           hdrfn=hdrf.name, outfn=sortedfn)
        #     subprocess.call(sortcmd, shell=True)
        #     ofns.append(sortedfn)
    except:
        raise
    finally:
        #os.remove(of.name)
        pass
    shutil.rmtree(odir1)
    return ofns

def preprocess(args):
    if (args.array_jobs is None and args.array_index is not None) or \
       (args.array_jobs is not None and args.array_index is None):
       sys.stderr.write('Both --array-jobs and --array-index must be specified\n')
       sys.exit(2)
    if args.output is None and (args.array_jobs is not None or args.array_index is not None):
        sys.stderr.write('--output must be specified when running a job array\n')
        sys.exit(2)
    if os.path.exists(args.output) and args.no_overwrite:
        sys.stderr.write('Output {} already exists - not overwriting\n'.format(args.output))
        sys.exit(0)
    mutation_code, reverse_mutation_code = read_codes(args.mutation_coding)
    if args.errors != '-':
        args.errf = open(args.errors, 'w')
    else:
        args.errf = open(os.devnull, 'w')

    global warned_invalid_chrom
    warned_invalid_chrom = False
    fns = []
    for ddir_or_fnlist in args.input:
        if os.path.isdir(ddir_or_fnlist):
            fns.extend(filter(accept_suffix,
                              map(lambda d: os.path.join(ddir_or_fnlist, d), os.listdir(ddir_or_fnlist))))
        else:
            fns.append(ddir_or_fnlist)
    n_missing = 0
    for fn in fns:
        if os.path.exists(fn) == False:
            sys.stderr.write('Input file {} not found\n'.format(fn))
            n_missing += 1
    if n_missing > 0:
        sys.exit(1)
    status('{} input files found'.format(len(fns)), args)
    if len(fns) == 0:
        sys.exit(1)

    if args.no_ref_preload == False and args.negative_generation_mode != MODE_NEG_AUGMENT:
        status('Reading reference... ', args, lf=False)
        ref_genome = read_reference(args.reference, args.verbose)
    else:
        ref_genome = None

    if args.generate_negatives > 0 and args.negative_generation_mode in [MODE_NEG_GENERATE, MODE_NEG_AUGMENT]:
        variant_to_neg_sample, sample_ix_to_id = prepare_negative_examples(fns, args)
        fns = generate_negatives(fns, variant_to_neg_sample, sample_ix_to_id, args) # augmented input files replace the original ones as input
        if args.negative_generation_mode == MODE_NEG_AUGMENT:
            status('All done for augmenting input files with negative examples.\nRun preprocess --{} next.\n'.format(MODE_NEG_PROCESS), args)
            status('Augmented files in {}'.format(os.path.dirname(fns[0])), args)
            sys.exit(0)

    # process variants
    if args.output is None:
        o = sys.stdout
    else:
        if args.array_jobs is not None:
            # process a subset of all input files
            ofn = '{}.{}.tsv.gz'.format(args.output, args.array_index)
            o = gzip.open(ofn, 'w')
            fns = list(np.array_split(fns, args.array_jobs)[args.array_index - 1])
        else:
            # process all input files
            ofn = args.output
            o = gzip.open(args.output, 'w')
        try:
            os.makedirs(os.path.dirname(ofn))
        except:
            pass

    digits = int(np.ceil(np.log10(len(fns))))
    fmt = '{:' + str(digits) + 'd}/{:' + str(digits) + 'd} {}: '
    if args.info_column:
        infotag = '\t{}'.format('\t'.join(map(str.lower, args.info_column)))
    else:
        infotag = ''
    status('Writing mutation sequences...', args)
    o.write('chrom\tpos\tref\talt\tsample\tseq{}\n'.format(infotag))
    for i, fn in enumerate(fns):
        f, sample_name = open_stream(fn)
        status(fmt.format(i + 1, len(fns), sample_name), args)
        vr = get_reader(f, args)
        process_input(vr, o, sample_name, ref_genome, args.context,
                      mutation_code, reverse_mutation_code, args)
        f.close()
    o.close()
    status('Output written to {}'.format(ofn), args)
    status('All done.', args)
