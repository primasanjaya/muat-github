import os, sys, datetime, gzip, operator
from itertools import imap
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARAMETERS_NAME = 'dmm_config.json'
DEFAULT_DATA_CONFIG_NAME = 'data_config.json'
DEFAULT_MUTATION_CODING_NAME = 'mutation_codes.tsv'
DEFAULT_RUN_CONFIG = os.path.join(SRC_DIR, 'data', DEFAULT_DATA_CONFIG_NAME)
DEFAULT_MUTATION_CODING = os.path.join(SRC_DIR, 'data', DEFAULT_MUTATION_CODING_NAME)
DEFAULT_MODEL = 'model.h5'
DEFAULT_WEIGHTS = 'weights.h5'
BATCH_SIZE = 100
SAMPLE_COL = 4 # in output of map
FIRST_FEATURE_COL = 7  # in output of map
DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_INITIAL_MODEL_FN = 'model.initial.h5'
DEFAULT_BEST_LOSS_MODEL_FN = 'model.best.loss.h5'
DEFAULT_BEST_VALLOSS_MODEL_FN = 'model.best.valloss.h5'
DEFAULT_FINAL_MODEL_FN = 'model.final.h5'

MODE_MAE = 'mae'      # reconstruct mutation+DNA sequence
MODE_DAE = 'dae'      # reconstruct DNA sequence
MODE_PREDICT_SEQ = 'predict_seq'         # predict mutation sequence from DNA sequence
MODE_PREDICT_MASKED = 'predict_masked'   # predict masked mutation sequence from DNA sequence
MODE_PREDICT_SAMPLE = 'predict_sample'   # predict whether sample is correctly paired to mutation sequence
MODES = [MODE_MAE, MODE_DAE, MODE_PREDICT_SEQ, MODE_PREDICT_MASKED, MODE_PREDICT_SAMPLE]

ENCODING_DNA = 'dna'                          # "AGCTCCATAG"
ENCODING_MUTATION = 'mutation'                # "AGCT[C>T][C>T]ATAG"
ENCODING_MUTATION_MASKED = 'mutation_masked'  # "NNNN[C>T][C>T]NNNN"
ENCODING_TYPES = [ENCODING_DNA, ENCODING_MUTATION, ENCODING_MUTATION_MASKED]

# mode -> (input, output)
DATA_ENCODING = {MODE_MAE : (ENCODING_MUTATION, ENCODING_MUTATION),
                 MODE_DAE : (ENCODING_DNA, ENCODING_DNA),
                 MODE_PREDICT_SEQ : (ENCODING_DNA, ENCODING_MUTATION),
                 MODE_PREDICT_MASKED : (ENCODING_DNA, ENCODING_MUTATION_MASKED)}

COVARIATE_NUMERIC = 'numeric'
COVARIATE_CATEGORICAL = 'categorical'

CONFIG_KEY_VARIANT_DATA = 'variant_data'
CONFIG_KEY_COVARIATES = 'covariates'
CONFIG_KEY_COLUMNS = 'columns'
CONFIG_KEY_NORMALIZE = 'normalize'
CONFIG_KEY_SEQ_COLUMN = 'seq_column'
CONFIG_KEY_AUX_DATA_COLUMN = 'aux_data_column'
CONFIG_KEY_WEIGHTS = 'weights'
VARIANT_COVARIATE_KEY_MISSING = 'NA'

COVARIATE_PRE_LATENT = 'pre'
COVARIATE_POST_LATENT = 'post'

dna_comp = {'A' : 'T', 'C' : 'G', 'G' : 'C', 'T' : 'A',
            'N' : 'N', '-' : '-', '+' : '+'}

SAMPLE_VAR_NAME = 'sample'

VARIANT_INPUT_COL_REF = 2
VARIANT_INPUT_COL_ALT = 3
VARIANT_INPUT_COL_SAMPLE = 4
VARIANT_INPUT_COL_SEQ = 5

VARIANT_TYPE_NEG = 'neg'
VARIANT_TYPE_SUB = 'sub'
VARIANT_TYPE_INDEL = 'indel'
VARIANT_TYPE_SV = 'sv'
VARIANT_TYPE_MEI = 'mei'

def get_timestamp():
    return int((datetime.datetime.now() - datetime.datetime.utcfromtimestamp(0)).total_seconds() * 1000.0)

def get_timestr():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def hamming(s, t):
    return sum(imap(operator.ne, s, t))

def opensf(f, mode='r'):
    if f is None or f == '-':
        if mode == 'r':
            return sys.stdin
        elif mode in 'aw':
            return sys.stdout
        else:
            assert(0) # invalid mode
    elif f.endswith('.gz'):
        return gzip.open(f, mode)
    else:
        return open(f, mode)

def __pick_ref_base(x, rcodes, replace_dash=True):
    b = rcodes.get(x)[0] # {x : (ref, alt)}
    if replace_dash and b == '-':    # '-' (=insertion) is not in mutation code, replace with 'N'
        return 'N'
    else:
        return b

def __pick_alt_base(x, rcodes, replace_dash=True):
    b = rcodes.get(x)[1] # {x : (ref, alt)}
    if replace_dash and b == '-':    # '-' (=deletion) is not in mutation code, replace with 'N'
        return 'N'
    else:
        return b

def mutation_to_ref(s, rcodes, replace_dash=True):
    return ''.join(__pick_ref_base(a, rcodes, replace_dash) for a in s)

def mutation_to_alt(s, rcodes, replace_dash=True):
    return ''.join(__pick_alt_base(a, rcodes, replace_dash) for a in s)

def get_variant_input_header(infn):
    with gzip.open(infn) as f:
        line = f.readline().strip()
        if line.startswith('chr') or line[0] == '#':
            return line
        else:
            return None

def is_nonmutation(refalt):
    return refalt[0] == refalt[1]

def nan_ind(x):
    if np.isnan(x):
        return np.array((1, 0))
    else:
        return np.array((0, x))

N_PAIR = ('N', 'N')

def convert_mutation_sequence(seq, encoding, seq_target_len,
                              code_to_index, code_to_refalt):
    '''Encode and trim the input mutation sequence.
    If mode==ENCODING_DNA, remove insertions and pad the sequence with 'N's to match original length.
    if mode==ENCODING_MUTATION_MASKED, replace all non-mutation symbols with 'N's.
    If seq_target_len is set, trim the sequence to this length.'''
    assert(encoding in ENCODING_TYPES)
    n = len(seq)
    assert(seq_target_len is None or n >= seq_target_len)
    if encoding == ENCODING_DNA:
        seq = map(lambda x: code_to_refalt.get(x, N_PAIR)[0], seq)
        # delete insertions ('-' chars) and pad with N's
        lhs, rhs = filter(lambda x: x != '-', seq[:n/2]), filter(lambda x: x != '-', seq[n/2:])
        if len(lhs) < n/2:
            lhs = ['N' for i in range(n/2 - len(lhs))] + lhs
        if len(rhs) < n/2:
            rhs = rhs + ['N' for i in range(n/2 - len(rhs))]
        seq = ''.join(lhs + rhs)
    elif encoding == ENCODING_MUTATION_MASKED:
        seq = map(lambda x: 'N' if is_nonmutation(code_to_refalt.get(x, N_PAIR)) else x, seq)
    if seq_target_len is not None:   # trim sequence if required
        seq = seq[(n-seq_target_len)/2:n-(n-seq_target_len)/2]
        assert(len(seq) == seq_target_len)
    return seq

def mutation_sequence_to_one_hot(seq, encoding, seq_target_len,
                                 code_to_index, code_to_refalt):
    '''Convert the input mutation sequence into an one-hot encoded representation.
    If mode==ENCODING_DNA, remove insertions and pad the sequence with 'N's to match original length.
    if mode==ENCODING_MUTATION_MASKED, replace all non-mutation symbols with 'N's.
    If seq_target_len is set, trim the sequence to this length.'''
    seq = convert_mutation_sequence(seq, encoding, seq_target_len,
                                    code_to_index, code_to_refalt)
    s = map(code_to_index.get, seq)
    n = len(s)
    x = np.zeros((n, 1, len(code_to_index)))
    x[np.arange(n), 0, s] = 1
    return x

def one_hot_to_seq(x, index_to_code):
    "Input is NxK array with 1-to-K encoding"
    v = np.array([i for i in range(x.shape[1])])
    # [NxK] x [Kx1]
    return ''.join(map(index_to_code.get, np.dot(x, v)))

def prediction_to_seq(x, index_to_code):
    "Input is NxK array with values in [0, 1]"
    return ''.join(map(index_to_code.get, np.argmax(x, axis=1)))

class VariantMetadata:
    """Store metadata for input variants and manage
    1) covariate dimensionality, 2) numeric covariate normalization and
    3) imputation of missing values.
    Numeric variables: mean and standard deviation
    Categorical variables: list of categories"""
    def __init__(self, path, config, n_sample=100000):
        self.covariates = []  # list of (covariate name, size, type) tuples
        self.categories = {}  # covariate -> category -> index
        var_config = config[CONFIG_KEY_VARIANT_DATA]
        self.seq_column = var_config[CONFIG_KEY_COLUMNS][var_config[CONFIG_KEY_SEQ_COLUMN]]
        self.cov_columns = dict(map(lambda x: (x, var_config[CONFIG_KEY_COLUMNS][x]), var_config[CONFIG_KEY_COVARIATES]))
        self.type_weights = config[CONFIG_KEY_WEIGHTS] if CONFIG_KEY_WEIGHTS in config else None
        if path is not None:
            self.means, self.stds = self.estimate_covariate_mean_std(path, var_config, n_sample)
        config_covariates = var_config[CONFIG_KEY_COVARIATES]
        for cov in sorted(config_covariates):
            col = self.cov_columns[cov]
            if config_covariates[cov] == COVARIATE_NUMERIC:
                self.covariates.append((cov, 2, COVARIATE_NUMERIC, col, COVARIATE_PRE_LATENT)) # size=1 for value, 1 for missingness flag
            elif type(config_covariates[cov]) == list:
                if VARIANT_COVARIATE_KEY_MISSING in config_covariates[cov]:
                    config_covariates[cov].remove(VARIANT_COVARIATE_KEY_MISSING)
                self.covariates.append((cov, len(config_covariates[cov]) + 1, COVARIATE_CATEGORICAL, col, COVARIATE_PRE_LATENT))
                self.categories[cov] = dict([(v, i) for i, v in enumerate([VARIANT_COVARIATE_KEY_MISSING] + config_covariates[cov])])
            else:
                raise Exception('Invalid covariate in config: {}'.format(cov))

    def parse_variant_input(self, s, encoding, seq_target_len,
                            code_to_index, code_to_refalt):
        '''Parse a line of variant input file and return a list
        [sequence_in_one_hot, covariate1, covariate2, ..., covariateK].

        Arguments:
        s -- a line of variant input
        columns -- a list of column indexes, where columns[0] is the input sequence
        encoding -- if 'dna', use nucleotide encoding; if 'mutation', use
                mutation+nucleotide encoding
        seq_target_len -- if set, trim the sequence to this length
        code_to_index -- mutation code to index map
        code_to_refalt -- mutation code to (ref, alt) map
        '''
        v = s.rstrip().split('\t')
        out = [mutation_sequence_to_one_hot(v[self.seq_column], encoding, seq_target_len,
                                            code_to_index, code_to_refalt)]
        out += self.__parse_covariates(v)
        weight = self.__get_sample_weight(v) if self.type_weights else 1.0
        return out, weight

    def __get_sample_weight(self, v):
        """Determine variant type from REF and ALT fields and return weight of variant type.
        Variant type weights must be defined in data configuration."""
        ref, alt = v[VARIANT_INPUT_COL_REF], v[VARIANT_INPUT_COL_ALT]
        if len(ref) == 0 and len(alt) == 0:
            vtype = VARIANT_TYPE_NEG
        elif alt.startswith('SV_') or alt.startswith('MEI_'):
            vtype = VARIANT_TYPE_SV
        elif alt.startswith('MEI_'):
            vtype = VARIANT_TYPE_MEI
        elif len(ref) != len(alt):
            vtype = VARIANT_TYPE_INDEL
        else:
            vtype = VARIANT_TYPE_SUB
        return self.type_weights[vtype]

    def __parse_covariates(self, input):
        """Parse covariates from input"""
        out = []
        for cov_name, cov_size, cov_type, col, insertion_point in sorted(self.covariates):
            x = input[col]
            if cov_type == COVARIATE_NUMERIC:
                x = float(x)
                if np.isnan(x) == False and cov_name in self.means:
                    x = (x - self.means[cov_name]) / self.stds[cov_name]
                out.append(nan_ind(x))
            elif cov_type == COVARIATE_CATEGORICAL:
                # to one-hot
                enc = np.zeros(cov_size)
                if x not in self.categories[cov_name]:
                    raise Exception('Unspecified value {} for covariate {}. Valid values are: {}'.format(x, cov_name, ','.join(self.categories[cov_name])))
                enc[self.categories[cov_name][x]] = 1
                out.append(enc)
            else:
                raise Exception("Unknown covariate type: {}".format(cov_type))
        return out

    def get_covariates(self):
        return self.covariates

    def estimate_covariate_mean_std(self, path, var_config, nrows):
        """Calculate mean and standard deviation for covariates to be normalized
        using the first n rows found in path."""
        if CONFIG_KEY_NORMALIZE in var_config:
            if get_variant_input_header(path) is None:
                raise Exception('Header not found in input {}'.format(path))
            else:
                header = 0
            norm_cols = var_config[CONFIG_KEY_NORMALIZE]
            X = pd.read_csv(path, sep='\t', nrows=nrows, engine='c', low_memory=False,
                            header=header, usecols=norm_cols)
            m, s = X.mean(), X.std()
            means = [(i, m[i]) for i in norm_cols]
            stds = [(i, s[i]) for i in norm_cols]
            return dict(means), dict(stds)
        else:
            return {}, {}

class AuxDataSource:
    '''Sample metadata'''
    KEY_MISSING = 'NA'
    def __init__(self, fn, config,
                 norm_continuous=True,
                 skip_missing=True,
                 drop_sample_covariates=False):   # If true, retain only sample id (used in mutation mapping)
        self.skip_missing = skip_missing
        if drop_sample_covariates:
            self.__drop_sample_covariates(config)
        # order of covariates must match with the model specification (models.py)
        covariates = sorted(config['aux_data']['covariates'])
        self.__aux_id_col = config['aux_data']['id_column']
        self.covariate_names = [x[0] if type(x) == list else x for x in covariates]
        if self.__aux_id_col in self.covariate_names:
            # sample identifier may be both aux_id and covariate;
            # in this case, it must be specified only once to avoid duplicate columns in __data
            select_cols = self.covariate_names
        else:
            select_cols = [self.__aux_id_col] + self.covariate_names
        self.__data = pd.read_csv(fn, sep='\t')[select_cols]
        self.__covariates = covariates
        self.__categories = {}  # covariate -> value -> index
        cov_info = []  # [(covariate, dimension, type), ...]
        # set up a lookup table for covariate dimensionality
        for cov in covariates:
            if type(cov) == list:  # name, model insertion point (pre/post)
                cov, insertion = cov[0], cov[1]
            else:
                insertion = COVARIATE_PRE_LATENT

            if (self.__data.dtypes[cov] == np.float64) or (self.__data.dtypes[cov] == np.int64):
                self.__categories[cov] = None
                cov_info.append((cov, 2, COVARIATE_NUMERIC, None, insertion))  # indicator of missingness + covariate
                if norm_continuous:
                    self.__data[cov] = (self.__data[cov] - np.nanmean(self.__data[cov])) / np.nanstd(self.__data[cov])
            elif self.__data.dtypes[cov] == np.object:
                cats = sorted(self.__data[cov].unique())
                if AuxDataSource.KEY_MISSING in cats:
                    cats.remove(AuxDataSource.KEY_MISSING)
                # NA=0, cat1=1, cat2=2, ...
                self.__categories[cov] = \
                    dict([(val, ix + 1) for ix, val in enumerate(cats)])
                self.__categories[cov][AuxDataSource.KEY_MISSING] = 0
                cov_info.append((cov, len(self.__categories[cov]), COVARIATE_CATEGORICAL, None, insertion))
            else:
                raise Exception('Unsupported data type {} in aux data'.format(self.__data.dtypes[cov]))
        #config['aux_data']['covariates'] = cov_info
        self.covariates = cov_info

        # set up the covariate arrays for get_covariates() to return
        self.__covariate_arrays = {}  # aux_id -> [cov1_array, cov2_array, ...]
        for i, r in self.__data.iterrows():
            C = []
            for cov in self.covariate_names:
                cats = self.__categories[cov]
                if cats is None:
                    C.append(nan_ind(r[cov]))
                else:
                    enc = np.zeros(len(cats))
                    enc[cats[r[cov]]] = 1
                    C.append(enc)     # categorical variable, one-hot encode
            if r[self.__aux_id_col] in self.__covariate_arrays:
                raise Exception('Aux data id {} not unique'.format(r[self.__aux_id_col]))
            self.__covariate_arrays[r[self.__aux_id_col]] = C
        # covariates for samples not found in aux data
        self.__nan_covariate_arrays = self.__get_nan_covariate_arrays()

    def get_covariates(self):
        return self.covariates

    def __get_nan_covariate_arrays(self):
        'Return all-NaN covariate arrays'
        C = []
        for cov in self.covariate_names:
            cats = self.__categories[cov]
            if cats is None:
                C.append(nan_ind(np.nan)) # indicator for missingness
            else:
                enc = np.zeros(len(cats))
                enc[0] = 1
                C.append(enc) # one-hot of 0 (=NA)
        return C

    def get_covariate_arrays(self, aux_id):
        '''Return a list of arrays containing covariate values for aux_id.
        '''
        if aux_id in self.__covariate_arrays:
            return self.__covariate_arrays[aux_id]
        else:
            if self.skip_missing:
                return None
            else:
                return self.__nan_covariate_arrays

    def __drop_sample_covariates(self, config):
        '''Remove all sample covariates except sample identity. This is used when
        retraining the sample latent mapping with new samples.'''
        c = dict(config['aux_data']['covariates'])
        print c
        if SAMPLE_VAR_NAME not in c:
            raise Exception('Sample id not in metadata')
        config['aux_data']['covariates'] = [[SAMPLE_VAR_NAME, c[SAMPLE_VAR_NAME]]]

def generate_one_hot_mutations_from_file(path, config, code_to_index, code_to_refalt,
                                         n_samples_in_batch, mode,
                                         input_seq_target_len, output_seq_target_len,
                                         variant_metadata, aux_data_source):
    """Return a tuple (inputs, outputs, weights), where
    * inputs[0] is a BxNxK array where B=batch size, N=sequence length and K=number of mutations
    * inputs[1:] are BxM covariate arrays, where M is specific to the covariate and given in config.
    * outputs are either 1) reconstructed inputs if in unsupervised mode, or
      2) a BxNxK array containing the predicted sequence if in supervised mode.
    * weights is a Bx1 array of sample weights

    Continuous variant-level covariate values are transformed to Z-scores, if
    requested in config.
    """
    # find column indexes for sequence and covariate inputs in variant input file
    var_data = config[CONFIG_KEY_VARIANT_DATA]
    # order of variables here needs to be the same as in the model specification (models.py)
    if aux_data_source is not None:
        aux_index_column = var_data[CONFIG_KEY_COLUMNS][var_data[CONFIG_KEY_AUX_DATA_COLUMN]]
    c = 0
    X, Y, W = [], [], []  # [input sequence, covariates], [target sequence]
    input_encoding, output_encoding = DATA_ENCODING[mode]
    while 1:
        f = gzip.open(path)
        for line in f:
            if line == '' or line[0] == '#' or line.startswith('chr'):
                continue
            data, weight = variant_metadata.parse_variant_input(\
                                line, input_encoding, input_seq_target_len,
                                code_to_index, code_to_refalt)
            v = line.strip().split('\t')
            if aux_data_source is not None:
                aux_cov = aux_data_source.get_covariate_arrays(v[aux_index_column])
                if aux_cov is None:  # missing sample in aux data
                    continue
                data.extend(aux_cov)
            X.append(data)
            W.append(weight)
            if input_encoding == output_encoding:
                Y.append(data)
            else:
                # supervised mode: output only the sequence as prediction target
                Y.append([mutation_sequence_to_one_hot(v[variant_metadata.seq_column],
                                                       output_encoding, output_seq_target_len,
                                                       code_to_index, code_to_refalt)])
            c += 1
            if c == n_samples_in_batch:
                X = np.array(X)  # [n_mutations x n_covariates]
                # reorganize to [seqs, cov1, cov2, ..., covK] format
                inputs = [np.reshape(np.concatenate(X[:, i]), \
                    (X.shape[0],) + X[:, i][0].shape) for i in xrange(X.shape[1])]
                Y = np.array(Y)
                outputs = [np.reshape(np.concatenate(Y[:, i]), \
                    (Y.shape[0],) + Y[:, i][0].shape) for i in xrange(Y.shape[1])]
                yield (inputs, outputs, np.array(W))
                c = 0
                X, Y, W = [], [], []
        f.close()

def permute(x):
    'Permute x in-place such that x[i] != y[i] if x[i] is unique'
    for i in range(len(x) - 1):
        j = np.random.randint(i + 1, len(x))
        x[i], x[j] = x[j], x[i]

def generate_mutation_sample_pairings(path, config, code_to_index, code_to_refalt,
                                      n_samples_in_batch, mode, seq_target_len,
                                      variant_metadata, aux_data_source, match_pairs=0.5):
    """Generate pairs of mutation-sample such that half of pairs are correctly
    matched and for the remaining half each mutation is randomly assigned an
    incorrect originating sample within the same batch.

    Return a tuple (inputs, outputs), where
    * inputs[0] is a BxNxK array where B=batch size, N=sequence length and K=number of mutations
    * inputs[1:] are BxM covariate arrays, where M is specific to the covariate and given in config.
    * outputs are binary labels whether the sequence and sample (i.e., sample-level covariates)
      are correctly matched.

    Continuous variant-level covariate values are transformed to Z-scores, if
    requested in config.
    """
    assert(match_pairs > 0 and match_pairs <= 1)
    raise NotImplementedException('variant_metadata not supported')
    if aux_data_source is None:
        raise Exception('Aux data must be provided with --mode=predict_sample')
    # find column indexes for sequence and covariate inputs in variant input file
    var_data = config[CONFIG_KEY_VARIANT_DATA]
    # order of variables here needs to be the same as in the model specification (models.py)
    variables = [var_data[CONFIG_KEY_SEQ_COLUMN]] + sorted(map(lambda x: x[0], var_data[CONFIG_KEY_COVARIATES]))
    try:
        columns = [var_data[CONFIG_KEY_COLUMNS][k] for k in variables] # e.g., ['seq', 'expr', 'meth']
    except KeyError:
        raise Exception('Covariate "{}" does not exist in data'.format(k))
    means, stds = estimate_covariate_mean_std(path, columns, var_data)
    aux_index_column = var_data[CONFIG_KEY_COLUMNS][var_data[CONFIG_KEY_AUX_DATA_COLUMN]]
    c = 0
    V, S = [], []  # variant data, sample data
    while 1:
        f = gzip.open(path)
        for line in f:
            if line == '' or line[0] == '#' or line.startswith('chr'):
                continue
            data = parse_variant_input(line, columns, ENCODING_MUTATION, seq_target_len,
                                       code_to_index, code_to_refalt, means, stds)
            V.append(data)
            v = line.strip().split('\t')
            S.append(aux_data_source.get_covariate_arrays(v[aux_index_column]))

            c += 1
            if c == int(n_samples_in_batch * match_pairs):
                # set of correctly paired sequences and samples (n == c)
                X = map(lambda x: x[0] + x[1], zip(V, S))
                # set of incorrecly paired sequences and samples (n == c)
                c0 = n_samples_in_batch - c
                if c0 > 0:
                    S0 = list(S)
                    permute(S0)
                    X0 = map(lambda x: x[0] + x[1], zip(V, S0))
                    X.extend(X0[:c0])
                X = np.array(X)  # (n_samples_in_batch, n_attributes)

                # reorganize to [seqs, cov1, cov2, ..., covK] format
                inputs = [np.reshape(np.concatenate(X[:, i]), \
                    (X.shape[0],) + X[:, i][0].shape) for i in xrange(X.shape[1])]
                if c0 > 0:
                    outputs = np.concatenate([np.ones(c), np.zeros(c)[:c0]])
                else:
                    outputs = np.ones(c)

                yield (inputs, outputs)
                c = 0
                V, S = [], []
        f.close()

def prepare_generator(path, config, code_to_index, code_to_refalt,
                      n_samples_in_batch, mode,
                      input_seq_target_len, output_seq_target_len,
                      variant_metadata, aux_data_source, **kwargs):
    if mode == MODE_PREDICT_SAMPLE:
        return generate_mutation_sample_pairings(path, config, code_to_index, code_to_refalt,
                                                 n_samples_in_batch, mode, input_seq_target_len,
                                                 variant_metadata, aux_data_source, **kwargs)
    elif mode in [MODE_MAE, MODE_DAE, MODE_PREDICT_SEQ, MODE_PREDICT_MASKED]:
        return generate_one_hot_mutations_from_file(path, config, code_to_index, code_to_refalt,
                                                    n_samples_in_batch, mode,
                                                    input_seq_target_len, output_seq_target_len,
                                                    variant_metadata, aux_data_source)
    else:
        raise ValueError('Unsupported mode: {}'.format(mode))

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

def init_codes(fn):
    "Return code_to_refalt, code_to_index, index_to_code"
    refalt_to_code, code_to_refalt = read_codes(fn)
    codes = sorted(code_to_refalt.keys())
    code_to_index = dict((v, k) for (k, v) in enumerate(codes))
    index_to_code = dict((k, v) for (k, v) in enumerate(codes))
    return code_to_refalt, code_to_index, index_to_code

def find_seq_len(infn, config):
    var_data = config['variant_data']
    ix = var_data['columns'][var_data['seq_column']]
    with gzip.open(infn) as f:
        line = f.readline()
        if line.startswith('chr') or line[0] == '#':  # input contains header
            line = f.readline()
        return len(line.strip().split('\t')[ix])

def ispowerof2(x):
    return x != 0 and (x & (x - 1)) == 0

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

def strip_suffixes(s, suffixes):
    for suffix in suffixes:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return s

def get_encoder_mask(variant_metadata, aux_metadata, args):
    '''Return a binary array where m[i] == 1 iff generated input i is input for the encoder.
Otherwise i is input for the decoder. The mask must match both the generator (common.py) and
the models (models.py).

Mask: seq, variant_covariates, sample_covariates.
'''
    m = []
    # variant covariates
    if args.reg_weight > 0:
        is_enc = 1  # sequence and variant covariates are input to the decoder
    else:
        is_enc = 0  # ...to the encoder
    m.append(is_enc)
    m.extend([is_enc for _ in variant_metadata.get_covariates()])

    # sample covariates
    for cov in aux_metadata.get_covariates():
        if cov[4] == COVARIATE_PRE_LATENT:
            m.append(1)
        else:
            assert(cov[4] == COVARIATE_POST_LATENT)
            m.append(0)

    return m

def update_args_with_params(args, model_params, overwrite=False, skip=[]):
    if type(model_params) != list:
        model_params = [model_params]
    for i, params in enumerate(model_params):
        for param in params:
            if param in skip:
                continue
            p = param.replace('-', '_')
            if overwrite or hasattr(args, p) == False or getattr(args, p) is None:
                setattr(args, p, params[param])

def read_lines(f, n):
    L = []
    c = 0
    for s in f:
        L.append(s.strip())
        c += 1
        if c == n:
            break
    return L, c
