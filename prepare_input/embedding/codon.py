import os
import sys
import math
import pandas as pd
from statistics import mean

def translate(codon_seq: str, codon2aa: dict):
    aaseq = ''
    codon2aa_href = ''
    
    codon_seq = codon_seq.upper()
    codon_seq = codon_seq.replace('U', 'T')
    for codon in SliceCodonStr(codon_seq):
        aa = ''
        if codon not in codon2aa:
            print(f'Codons {codon} cannot be translated.', file=sys.stderr)
            exit(0)
        else:
            aa = codon2aa[codon]
        aaseq += aa
    
    return aaseq


def set_codon2aa():
    d = {}
    d['TTT'] = 'F'
    d['TTC'] = 'F'
    d['TTA'] = 'L'
    d['TTG'] = 'L'
    d['CTT'] = 'L'
    d['CTC'] = 'L'
    d['CTA'] = 'L'
    d['CTG'] = 'L'
    d['ATT'] = 'I'
    d['ATC'] = 'I'
    d['ATA'] = 'I'
    d['ATG'] = 'M'
    d['GTT'] = 'V'
    d['GTC'] = 'V'
    d['GTA'] = 'V'
    d['GTG'] = 'V'
    d['TCT'] = 'S'
    d['TCC'] = 'S'
    d['TCA'] = 'S'
    d['TCG'] = 'S'
    d['CCT'] = 'P'
    d['CCC'] = 'P'
    d['CCA'] = 'P'
    d['CCG'] = 'P'
    d['ACT'] = 'T'
    d['ACC'] = 'T'
    d['ACA'] = 'T'
    d['ACG'] = 'T'
    d['GCT'] = 'A'
    d['GCC'] = 'A'
    d['GCA'] = 'A'
    d['GCG'] = 'A'
    d['TAT'] = 'Y'
    d['TAC'] = 'Y'
    d['TAA'] = '*'
    d['TAG'] = '*'
    d['TGA'] = '*'
    d['CAT'] = 'H'
    d['CAC'] = 'H'
    d['CAA'] = 'Q'
    d['CAG'] = 'Q'
    d['AAT'] = 'N'
    d['AAC'] = 'N'
    d['AAA'] = 'K'
    d['AAG'] = 'K'
    d['GAT'] = 'D'
    d['GAC'] = 'D'
    d['GAA'] = 'E'
    d['GAG'] = 'E'
    d['TGT'] = 'C'
    d['TGC'] = 'C'
    d['TGG'] = 'W'
    d['CGT'] = 'R'
    d['CGC'] = 'R'
    d['CGA'] = 'R'
    d['CGG'] = 'R'
    d['AGT'] = 'S'
    d['AGC'] = 'S'
    d['AGA'] = 'R'
    d['AGG'] = 'R'
    d['GGT'] = 'G'
    d['GGC'] = 'G'
    d['GGA'] = 'G'
    d['GGG'] = 'G'

    return d

def SliceCodonStr(orf):
    return [orf[i: i+3] for i in range(0, len(orf), 3)]

