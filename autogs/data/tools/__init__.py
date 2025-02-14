"""Script to read VCF files"""
 
import numpy as np
import pandas as pd

def genotype_to_numeric(genotype):
    if genotype in ["0/0", "0|0"]:
        return (0.0-1)
    elif genotype in ["0/1", "1/0", "0|1", "1|0"]:
        return (1.0-1)
    elif genotype in ["1/1", "1|1"]:
        return (2.0-1)
    elif genotype in ["./.", ".|."]:
        return np.nan 
    else:
        raise ValueError(f"Unexpected genotype format: {genotype}")

class Record(object):
    def __init__(self, line, sample_names):
        info = line.strip().split("\t")
        self.CHROM  = info[0]
        self.POS    = info[1]
        self.ID     = info[2]
        self.REF    = info[3]
        self.ALT    = info[4]
        self.QUAL   = info[5]
        self.FILTER = info[6]
        self.INFO   = info[7] 
        self.FORMAT = info[8] 
        self.sample_genotypes = {sample_name: genotype_to_numeric(genotype) 
                                 for sample_name, genotype in zip(sample_names, info[9:])}

class rVCF(object):
    def __init__(self, file_path):
        self.header = []
        self.sample_names = []
        self.reader = open(file_path, 'r')
        self.line = None 

        while self.line is None or self.line.startswith('#'):
            if self.line is not None:
                self.header.append(self.line)
                if self.line.startswith('#CHROM'):
                    self.sample_names = self.line.split('\t')[9:]
            self.line = self.reader.readline().strip()

    def __iter__(self): 
        return self 

    def __next__(self): 
        while self.line.startswith('#'):  
            self.line = self.reader.readline().strip()

        if self.line != "":
            record = Record(self.line, self.sample_names)
            self.line = self.reader.readline().strip() 
            return record
        else:
            self.reader.close()
            raise StopIteration()

    def reader_close(self):
        self.reader.close()

class VCF(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.rvcf = rVCF(file_path)

    def parse_records(self):
        records = []
        for record in self.rvcf:
            records.append([
                record.CHROM,
                record.POS,
                record.ID,
                record.REF,
                record.ALT,
                record.QUAL,
                record.FILTER,
                record.INFO,
                record.FORMAT,
                *record.sample_genotypes.values()
            ])
        return pd.DataFrame(records, columns=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + list(self.rvcf.sample_names))