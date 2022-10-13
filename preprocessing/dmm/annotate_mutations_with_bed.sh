#!/bin/bash
#
# Annotate a mutation file with the 5th column of a BED file.
#
# Note: header is expected in both inputs.
#

if [[ -z "$3" ]]; then
  echo "Usage: $0 mutations annotations output [column_name]"
  exit 2
fi

if [[ -z "$4" ]]; then
  label=$(basename ${2%.bed.gz})
else
  label=$4
fi

set -euo pipefail

muttsv=$1
annbed=$2
outtsv=$3

tab=$'\t'
IFS=" "  # preserve tabs in header
echo "Extracting header from ${muttsv} ..."
hdr=$(cat <(zcat ${muttsv}|head -n 1) )
hdr="${hdr}${tab}${label}"
echo "Annotating ${muttsv} with ${annbed} and writing to ${outtsv} ..."
date
# write header + input in TSV format with mean of annotation overlap for each mutation
cat <(echo ${hdr}) <(bedmap --sweep-all --delim '\t' --bp-ovr 1 --faster --echo --mean \
  <(gunzip -c ${muttsv}|tail -n +2|awk 'BEGIN{FS=OFS="\t"} {$2 = $2 OFS $2+1} 1') <(gunzip -c ${annbed}|tail -n +2) | sed 's/NAN/nan/' | cut -f 1-2,4-) | bgzip -c >${outtsv}
echo "All done"
date
