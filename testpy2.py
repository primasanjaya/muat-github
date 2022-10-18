import subprocess

if __name__ == '__main__':
        
        '''
        subprocess.call("python2 preprocessing/dmm/annotate_mutations_with_gc_content.py \
        -i /mnt/g/experiment/muat/data/raw/temp/00b9d0e6-69dc-4345-bffd-ce32880c8eef.consensus.20160830.somatic.snv_mnv.tsv.gz \
        -o /mnt/g/experiment/muat/data/raw/temp/00b9d0e6-69dc-4345-bffd-ce32880c8eef.consensus.20160830.somatic.snv_mnv.gc.tsv.gz \
        -n 1001 \
        -l gc1kb \
        --reference /mnt/g/experiment/muat/hs37d5_1000GP.fa \
        --verbose", shell=True)

        pdb.set_trace()
        
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
        '''

        import os
        from liftover import get_lifter
        import pdb

        converter = get_lifter('hg38', 'hg19')
        chrom = '1'
        pos = 103786442
        converter[chrom][pos]

        # other synonyms for the lift call
        converter.convert_coordinate(chrom, pos)
        






        