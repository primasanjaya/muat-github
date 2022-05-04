#!/usr/bin/env python2
'''
Deep neural network models of (somatic) mutagenesis.
'''
import sys, os, argparse, random, json, tempfile
import numpy as np

from common import *
import model

DEFAULTS_JSON = os.path.join(os.path.abspath(os.path.dirname(__file__)), DEFAULT_PARAMETERS_NAME)
if os.path.exists(DEFAULTS_JSON) == False:
    sys.stderr.write('Warning: default parameter file missing ({})'.format(DEFAULTS_JSON))
    defaults = {}
else:
    defaults = json.load(open(DEFAULTS_JSON))

def cmd_preprocess(args):
    if ispowerof2(args.context) == False:
        sys.stderr.write('Error: --context must be a power of 2\n')
        sys.exit(2)
    import preprocess
    preprocess.preprocess(args)

def cmd_train(args):
    if args.overwrite and args.continue_training:
        sys.stderr.write('Error: specify either --overwrite or --continue-training, not both\n')
        sys.exit(2)
    if args.mode not in MODES:
        sys.stderr.write('Error: mode should be one of {}\n'.format(', '.join(MODES)))
        sys.exit(2)
    if args.mode in [MODE_PREDICT_SEQ, MODE_PREDICT_MASKED] and args.trim is None:
        sys.stderr.write('''Error: --trim needs to be set in predictive modes.
Suggested value is n/2 where n is input sequence length.
Trimming is necessary to make sure the input sequence does not contain information about insertions.\n''')
        sys.exit(2)
    if args.trim is not None and (np.log2(args.trim) % 1) != 0:
        sys.stderr.write('Error: --trim must be a power of 2\n')
        sys.exit(2)
    import train
    train.train(args)

def cmd_map(args):
    import mapm
    mapm.map_mutations(args)

def cmd_plot(args):
    import plot
    plot.plot(args)

def cmd_plotmodel(args):
    import plotmodel
    plotmodel.plotmodel(args)

def cmd_generate(args):
    import generate
    if args.latent_vars is not None:
        if args.latent_vars == 'all':
            args.latent_vars = None
        args.latent_vars = map(int, args.latent_vars.split(','))
    generate.generate(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    subparsers = p.add_subparsers(dest="cmd")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('-v', '--verbose', type=int, help='Try to be more verbose')
    common_parser.add_argument('--mutation-coding', help='Mutation coding table ("ref alt code"/line) [{}]'.format(\
                               defaults['mutation_coding']), metavar='fn')
    common_parser.add_argument('--config', help='Read parameters from a JSON file')
    common_parser.add_argument('--data-config',
                               help='Column specification for --input, --validation and --aux-data  [{}]'.format(\
                               defaults['data_config']))
    common_parser.add_argument('--random-seed', default=None, type=int, metavar='seed')
    common_parser.add_argument('--tmp')

    prep_parser = subparsers.add_parser('preprocess', parents=[common_parser],
                                        help='Prepare mutation data for training')
    prep_parser.add_argument('-i', '--input', action='append', metavar='dir(s)',
                             help='Either a directory with vcf/maf[.gz] files or a vcf/maf[.gz] file (-i may be given more than once)', required=True)
    prep_parser.add_argument('-o', '--output', metavar='fn', help='Preprocessed mutation data')
    prep_parser.add_argument('-r', '--reference', metavar='ref', help='Reference genome (fasta) [{}]'.format(\
                             defaults['reference']))
    prep_parser.add_argument('-k', '--context', help='Sequence context length (power of 2) [{}]'.format(\
                             defaults['context']), metavar='bp', type=int)
    prep_parser.add_argument('-e', '--errors', metavar='fn',
                             help='File where to log errors [{}]'.format(defaults['errors']))
    prep_parser.add_argument('--no-ref-preload', help='Use samtools to read reference on demand (slow but fast startup) [false]',
                             action='store_true')
    prep_parser.add_argument('--no-filter', help='Process all variants [default=only PASS/. variants]',
                             action='store_true')
    prep_parser.add_argument('--sample-id', help='Sample identifier column name in MAF file')
    prep_parser.add_argument('-n', '--generate_negatives', help='Ratio of negative to positive examples [{}]. Two passes on data are required for n>0.'.format(\
                             defaults['negative_ratio']), type=float)
    prep_parser.add_argument('--median-variant-type-negatives', action='store_true',
                             help='Generate median number of each variant type as negative examples for each sample')
    prep_parser.add_argument('--median-variant-type-file', help='Load median variant numbers from a file')
    prep_parser.add_argument('--negative-generation-mode', help='[generate] output in one go (default), [augment] input files or [process] augmented files', default='generate')
    prep_parser.add_argument('--info-column', help='Input column name to write toutputo output (MAF input only). May be specified more than once.', action='append')
    prep_parser.add_argument('--report-interval', help='Interval to report number of variants processed',
                             type=int)
    prep_parser.add_argument('--array-jobs', help='How many array jobs in total', type=int)
    prep_parser.add_argument('--array-index', help='Index of this job', type=int)
    prep_parser.add_argument('--nope', help='Only one variant per output sequence', action='store_true')
    prep_parser.add_argument('--no-overwrite', help='Do not overwrite if output exists', action='store_true')
    prep_parser.set_defaults(func=cmd_preprocess)

    train_parser = subparsers.add_parser('train', parents=[common_parser],
                                         help='Train a mutation autoencoder model')
    train_parser.add_argument('-i', '--input', metavar='fn',
                              help='Preprocessed mutation data - training set')
    train_parser.add_argument('-l', '--validation', metavar='fn',
                              help='Preprocessed mutation data - validation set. Use training data if not specified.')
    train_parser.add_argument('-a', '--aux-data', metavar='fn',
                              help='Auxiliary data containing covariates')
    train_parser.add_argument('-m', '--model', metavar='dir',
                              help='Where to write trained models and statistics.')
    train_parser.add_argument('-d', '--decay', help='Decay factor of loss weights [{}]'.format(\
                              defaults['decay']), type=float, metavar='float')
    train_parser.add_argument('-e', '--epochs', help='How many epochs to train [{}]'.format(\
                              defaults['epochs']), type=int, metavar='int')
    train_parser.add_argument('-n', '--train-epoch-size', help='How many training samples per epoch [{}]'.format(\
                              defaults['train_epoch_size']), type=int, metavar='int')
    train_parser.add_argument('--validation-epoch-size',
                              type=int, metavar='int', help='How many validation samples to use [{}]'.format(\
                              defaults['validation_epoch_size']))
    train_parser.add_argument('--mode',
                              help='''Mode of operation. \n
                                      {}: Mutation sequence autoencoder (input/output=mutation sequence), \
                                      {}: DNA sequence autoencoder (input/output=DNA sequence),\n \
                                      {}: Predict mutation sequence from DNA sequence (input=DNA,output=mutation sequence),\n \
                                      {}: Predict masked mutation sequence from DNA sequence (input=DNA,output=mutation sequence with nonmutations replaced with Ns),\n \
                                      {}: Predict sample from mutation sequence (input=mutation sequence, output=sample)
                                      [{}]'''.\
                                      format(MODE_MAE, MODE_DAE, MODE_PREDICT_SEQ, MODE_PREDICT_MASKED, MODE_PREDICT_SAMPLE,
                                             MODE_PREDICT_MASKED))
    train_parser.add_argument('--trim', default=None, type=int,
                              help='''Trim input sequences to N bp. Trimming to half of input sequence length is recommended in
                                      predictive modes [no trim]''')
    train_parser.add_argument('--output-seq-len', help='Length of the predicted sequence [{}]'.format(defaults['output_seq_len']), type=int)
    train_parser.add_argument('--batch-size', help='Batch size [{}]'.format(defaults['batch_size']), type=int)
    train_parser.add_argument('--latent-dim', help='Number of latent variables [{}]'.format(defaults['latent_dim']), type=int)
    train_parser.add_argument('--sample-latent-dim', help='Number of sample-level latent variables (only in --mode=hierarchical) [{}]'.format(defaults['sample_latent_dim']), type=int)
    train_parser.add_argument('--filters', help='Number of filters [{}]'.format(defaults['filters']), type=int)
    train_parser.add_argument('--conv-length', help='Length of convolution in bp [{}]'.format(defaults['conv_length']), type=int)
    train_parser.add_argument('--conv-depth', help='Number of convolutional layers [log2(sequence length)]', type=int)
    train_parser.add_argument('--intermediate-depth', help='Number of fully connected layers [{}]'.format(defaults['intermediate_depth']), type=int)
    train_parser.add_argument('--intermediate-dim', help='Dimensionality of fully connected layers [{}]'.format(defaults['intermediate_dim']), type=int)
    train_parser.add_argument('--sample-intermediate-dim', help='Dimensionality of fully connected layers in sample-specific module (only in --mode=hierarchical) [{}]'.format(defaults['sample_intermediate_dim']), type=int)
    train_parser.add_argument('--sample-intermediate-depth', help='Number of fully connected layers in sample-specific module (only in --mode=hierarchical) [{}]'.format(defaults['sample_intermediate_depth']), type=int)
    train_parser.add_argument('--prior-variance', help='Variance of the Gaussian prior [{}]'.format(defaults['prior_variance']), type=float)
    train_parser.add_argument('--regularization', help='Regularization method (l1, mmd) [{}]'.format(defaults['regularization']))
    train_parser.add_argument('--reg-weight', help='Weight of variant regularization term [{}]'.format(defaults['reg_weight']), type=float)
    train_parser.add_argument('--sample-reg-weight', help='Weight of sample regularization term [{}]'.format(defaults['sample_reg_weight']), type=float)
    train_parser.add_argument('--distribution', help='Latent variable distribution (only in MMD regularization) (gaussian, gamma, zero) [{}]'.format(defaults['distribution']))
    train_parser.add_argument('--gamma-shape', type=float, help='Gamma shape parameter with --distribution=gamma [{}]'.format(defaults['gamma_shape']))
    train_parser.add_argument('--gamma-scale', type=float, help='Gamma scale parameter with --distribution=gamma [{}]'.format(defaults['gamma_scale']))
    train_parser.add_argument('--learning-rate', help='Learning rate [{:.2e}]'.format(defaults['learning_rate']), type=float)
    train_parser.add_argument('--dropout', help='Dropout rate [{}]'.format(defaults['dropout']), type=float)
    train_parser.add_argument('--overwrite', help='If --model already exists, start learning from scratch instead of continuing',
                              action='store_true')
    #train_parser.add_argument('--arch', help='Architecture to use (gaussian, hierarchical, predict_sample, bernoulli) [%(default)s]',
    #                          default='gaussian')
    train_parser.add_argument('--save-epoch-models', help='Save the model after each training epoch',
                              action='store_true')
    train_parser.add_argument('--control-z-ratio', help='Stop training if ratio of max(var(z))/min(var(z)) exceeds value [%(default)f]',
                              default=100.0, type=float)
    train_parser.add_argument('--early-stopping-delta', help='Stop training if val_loss decrease is below value. Negative values disable early stopping [{}]'.format(\
                              defaults['early_stopping_delta']), type=float)
    train_parser.add_argument('--early-stopping-patience', help='Wait for N epochs before stopping early [{}]'.format(\
                              defaults['early_stopping_patience']), type=int, metavar='N')
    train_parser.set_defaults(func=cmd_train)

    map_parser = subparsers.add_parser('map', parents=[common_parser],
                                       help='Map mutations to the latent space defined by the trained model')
    map_parser.add_argument('-i', '--input', metavar='fn', help='Preprocessed mutation data')
    map_parser.add_argument('-o', '--output', metavar='out',
                            help='Gzipped file with rows=mutations, columns=latent features')
    map_parser.add_argument('-m', '--model', required=True, metavar='dir',
                            help='Where to find the trained model.')
    map_parser.add_argument('-f', '--model-name', metavar='dir', default=DEFAULT_BEST_VALLOSS_MODEL_FN,
                            help='Model filename [{}]'.format(DEFAULT_BEST_VALLOSS_MODEL_FN))
    map_parser.add_argument('-n', '--n-mutations', help='Mutations to write [all]', type=int)
    map_parser.add_argument('--activations', help='Write filter activations', action='store_true')
    map_parser.add_argument('--mid-len', help='Flank size (bp) to display', default=5, type=int)
    map_parser.add_argument('--retrain-mapper', help='Retrain the sample mapping even if the model is compatible with input data [use original model if possible]',
                            action='store_true')
    map_parser.add_argument('--force-retrain', help='Retrain the sample mapping even if one already exists',
                            action='store_true')
    map_parser.add_argument('--retrain-dir', help='Where to store the retrained model [{}]'.format(\
                            defaults['retrain_dir']))
    map_parser.set_defaults(func=cmd_map)

    plot_parser = subparsers.add_parser('plot', parents=[common_parser],
                                        help='Compute and plot t-SNE features for mapped mutations')
    plot_parser.add_argument('-i', '--input', help='Output from dmm.py map', metavar='fn', required=True)
    plot_parser.add_argument('-o', '--output', help='Directory where to write results', metavar='dir', required=True)
    plot_parser.add_argument('--no-split', help='Plot all samples into the same figure [false]', action='store_true')
    plot_parser.add_argument('-n', '--components', help='Number of t-SNE components [%(default)d]',
                             default=2, type=int, metavar='int')
    plot_parser.add_argument('-p', '--perplexity', help='t-SNE perplexity [%(default)s]',
                             default=100.0, type=float, metavar='float')
    plot_parser.add_argument('--iterations', help='t-SNE iterations [%(default)d]',
                             default=1000, type=int, metavar='int')
    plot_parser.add_argument('--downsample', help='Downsample to N mutations for t-SNE [%(default)d]',
                             default=5000, type=int)
    plot_parser.set_defaults(func=cmd_plot)

    plotmodel_parser = subparsers.add_parser('plot-model', parents=[common_parser],
                                             help='Plot model characteristics (filters)')
    plotmodel_parser.add_argument('-m', '--model', help='Model directory', required=True, metavar='dir')
    plotmodel_parser.add_argument('-f', '--model-name', metavar='dir',
                                  help='Model filename [%(default)s]', default=DEFAULT_BEST_VALLOSS_MODEL_FN)
    plotmodel_parser.set_defaults(func=cmd_plotmodel)

    gen_parser = subparsers.add_parser('generate', parents=[common_parser],
                                       help='Generate representative mutations')
    gen_parser.add_argument('-i', '--input', required=True, metavar='fn',
                            help='Preprocessed mutation data')
    gen_parser.add_argument('-m', '--model', required=True, metavar='dir',
                            help='Where to find the trained model.')
    gen_parser.add_argument('-f', '--model-name', metavar='dir',
                            help='Model filename [%(default)s]', default=DEFAULT_BEST_VALLOSS_MODEL_FN)
    gen_parser.add_argument('-o', '--output', help='Where to write the results', metavar='fn')
    gen_parser.add_argument('-l', '--latent-vars', help='Latent variables to plot, comma-separated list [all]', metavar='N')
    gen_parser.add_argument('-n', '--n-mutations', help='Mutations to write [all]', type=int, metavar='N')
    gen_parser.add_argument('--deviation', help='Deviation(s) (%%) to generate, comma-separated if multiple [%(default)s]', default='33,67,100')
    gen_parser.set_defaults(func=cmd_generate)

    args = p.parse_args()
    if args.func == cmd_map and args.config is None:
        args.config = os.path.join(args.model, DEFAULT_PARAMETERS_NAME)
#    sys.stderr.write('*** ARGS from command line: {}\n'.format(args))
    if args.config:
        sys.stderr.write('Reading config from {}...\n'.format(args.config))
        params = [json.load(open(args.config)), defaults]
    else:
        sys.stderr.write('Reading default config from {}...\n'.format(DEFAULTS_JSON))
        params = [defaults]

    # use parameter from command line > config.json > defaults
    update_args_with_params(args, params, overwrite=False)
#    sys.stderr.write('*** ARGS after update {}\n'.format(args))
    if args.input is None:
        sys.stderr.write('--input is required\n')
        sys.exit(2)

    #if args.model is None:
    #    sys.stderr.write('--model is required\n')
    #    sys.exit(2)

    if args.tmp == 'None' or args.tmp == '':
        args.tmp = tempfile.gettempdir()
    if args.verbose:
        args.verbose = 1
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=all, 1=no info, 2=no warn, 3=no err
    if os.path.isabs(args.mutation_coding) == False and os.path.exists(args.mutation_coding) == False:
        args.mutation_coding = os.path.join(os.path.dirname(__file__), args.mutation_coding)
    if args.random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
    args.func(args)
