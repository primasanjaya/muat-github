import sys, gzip

# translation table to map each character to a nucleotide or N
valid_dna = ''.join([chr(x) if chr(x) in 'ACGTN' else 'N' for x in range(256)])

def openz(path, mode='r'):
    if path.endswith('.gz'):
        return gzip.open(path, mode)
    elif path == '-':
        if mode == 'r':
            return sys.stdin
        else:
            return sys.stdout
    else:
        return open(path, mode)

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

