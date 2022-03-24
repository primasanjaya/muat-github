#!/bin/bash
# This script takes preprocessed mutation data from 'dmm preprocess'
# and annotates it with various information:
#   - GC content
#   - Genic position (+/-)
#   - Exonic position (+/-)
#   - Orientation of the somatic mutation with respect to the coding strand orientation

if [[ -z "$1" ]]; then
  echo "Usage: $0 mutation-data"
  exit 2
fi

set -euo pipefail

if ! command -v bedops &> /dev/null
then
    echo "bedops not found: install with 'conda install bedops'"
    exit
fi

INFN=${1}  # input: preprocessed mutation data (output of 'dmm preprocess')
INFN2=${2} #folder
GCLEN=1001   # GC content window size

TDIR=$(mktemp -d -p .)
BASE=$(basename ${INFN} .gz)
BASE=${BASE%".tsv"}
SDIR=/path/to/projectdir/dmm2/scripts
DDIR=/path/to/projectdir/preprocessing/genomic_tracks/h37
REF=/path/to/genome_reference
DATADIR=/path/to/icgcdatadir/

echo "Reading input from ${INFN}"
echo "Writing temporary files to ${TDIR}"
echo "Output filename base: ${BASE}"

LC_ALL=C

# Sort input by sample, chromosome and position
echo Sorting input
cat <(zcat ${INFN} | head -1) <(zcat ${INFN} | 
  tail -n +2 | \
  sort -t$'\t' -k1,1 -k2g,2) | \
  bgzip -c > ${TDIR}/${BASE}.sorted.tsv.gz

# GC content 1 kb windows; ~5 h 10 min
echo GC content
/usr/bin/time -v ${SDIR}/annotate_mutations_with_gc_content.py \
  -i ${TDIR}/${BASE}.sorted.tsv.gz \
  -o ${TDIR}/${BASE}.gc.tsv.gz \
  -n ${GCLEN} \
  -l gc1kb \
  --reference ${REF} \
  --verbose

# Genic regions; ~30 min
echo Genic regions
/usr/bin/time -v ${SDIR}/annotate_mutations_with_bed.sh \
  ${TDIR}/${BASE}.gc.tsv.gz \
  ${DDIR}/Homo_sapiens.GRCh37.87.genic.genomic.bed.gz \
  ${TDIR}/${BASE}.gc.genic.tsv.gz \
  genic

# ~30 min
echo Exons
/usr/bin/time -v ${SDIR}/annotate_mutations_with_bed.sh \
  ${TDIR}/${BASE}.gc.genic.tsv.gz \
  ${DDIR}/Homo_sapiens.GRCh37.87.exons.genomic.bed.gz \
  ${TDIR}/${BASE}.gc.genic.exonic.tsv.gz \
  exonic

# Annotate dataset with gene orientation information
echo Transcript orientation
/usr/bin/time -v ${SDIR}/annotate_mutations_with_coding_strand.py \
  -i ${TDIR}/${BASE}.gc.genic.exonic.tsv.gz \
  -o ${TDIR}/${BASE}.gc.genic.exonic.cs.tsv.gz \
  --annotation ${DDIR}/Homo_sapiens.GRCh37.87.transcript_directionality.bed.gz \
  --ref ${REF}

# Replication timing / PCAWG consensus; ~30 min
#echo Replication timing
#/usr/bin/time -v ${SDIR}/annotate_mutations_with_bed.sh \
#   ${TDIR}/${BASE}.gc.genic.exonic.cs.tsv.gz \
#   ${DDIR}/merged_wavesignal_uw_repliseq_medians_GRCh37.nochr.rt.bed.gz \
#   ${TDIR}/${BASE}.gc.genic.exonic.cs.rt.tsv.gz \
#   rt

# Replication timing slope / PCAWG consensus; ~30 min
#echo Replication timing
#/usr/bin/time -v ${SDIR}/annotate_mutations_with_bed.sh \
#   ${TDIR}/${BASE}.gc.genic.exonic.cs.rt.tsv.gz \
#   ${DDIR}/merged_wavesignal_uw_repliseq_medians_GRCh37.nochr.rt_diff.bed.gz \
#   ${TDIR}/${BASE}.gc.genic.exonic.cs.rt.rtd.tsv.gz \
#   rt_diff

rsync ${TDIR}/${BASE}.gc.genic.exonic.cs.tsv.gz ${DATADIR}/${INFN2}/${BASE}.annotated.tsv.gz
echo "Annotation done. Result: ${BASE}.annotated.tsv.gz"
rm -r ${TDIR}
