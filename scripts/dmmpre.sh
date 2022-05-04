#!/bin/bash

CONTEXTLEN=256

if [[ -z "${1}" ]]; then
  echo "usage: $0 DIR_WITH_MAFS"
  exit 2
fi

echo "Process dir: ${1}"

pushd ${1}

ID=$(echo simple_somatic*tsv.gz | sed -rn 's/.*\.open\.(\S+).tsv.gz/\1/p')
echo "${ID}"

if [[ ! -s "simple_somatic_mutation.open.${ID}.tsv.gz" ]]; then
  echo "Somatic mutation file empty - skipping"
  exit
fi

cat \
  <(zcat simple_somatic_mutation.open.${ID}.tsv.gz|head -1) \
  <(zcat simple_somatic_mutation.open.${ID}.tsv.gz|tail -n +2|\
    LC_ALL=C sort -t$'\t' -u -k7,7 -k9,9 -k10g,10) | \
  bgzip -c > somatic_mutations.${ID}.unique.maf.tsv.gz

/path/to/projectdir/muat/dmm2/dmm.py preprocess \
  --mutation-coding /path/to/projectdir/dmm2/data/mutation_codes_sv.tsv \
  -i somatic_mutations.${ID}.unique.maf.tsv.gz \
  -o somatic_mutations.${ID}.dmm.k${CONTEXTLEN}.tsv.gz \
  -r /path/to/genome_reference/ \
  -n 1 \
  -k ${CONTEXTLEN} \
  --sample-id submitted_sample_id \
  --tmp /path/to/temp \
  -v 1 \


echo "Completed ${1}"

popd ${1}
