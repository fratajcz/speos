import pandas as pd
from typing import Iterable, Union



class LDBlockChecker:
    def __init__(self, ldblockfile="/mnt/storage/prs/ldblocks/EUR_LD_blocks_headless.bed",
                 genefile="/mnt/storage/prs/ldblocks/GRCh38_110_genes.tsv",
                 snpfile="/mnt/storage/prs/ldblocks/Liu_UC_snps.tsv",
                 snp_is_bed=True):
        """
            All input files must be in proper BED format, i.e. first column is "chr" + chromosome number (i.e. chr1), 
            second and third columns are base coordinates (begin, end), tabstop seperated. for snpfile base coordinates, begin and end are identical

            if snp_is_bed it assumes that the snpfile is one column of gene HGNC identifiers
        """
        self.ldblockfile = ldblockfile
        self.genefile = genefile
        self.snpfile = snpfile
        self.snp_is_bed = snp_is_bed

    def build_ldblocks(self):
        ldblocks = []

        file = pd.read_csv(self.ldblockfile, sep="\t", header=None)

        for i, row in file.iterrows():
            ldblocks.append(GenomicRange(int(row[0][3:]), row[1], row[2]))

        print("number of LD blocks: {}".format(len(ldblocks)))

        self.ldblocks = GenomicRangeContainer(ldblocks)

    def build_genes(self):
        genes = []

        file = pd.read_csv(self.genefile, sep="\t", header=None)

        for i, row in file.iterrows():
            genes.append(Gene(int(row[0][3:]), row[1], row[2], row[3]))

        print("number of genes: {}".format(len(genes)))

        self.genes = GenomicRangeContainer(genes)

    def build_coregenes(self, resultsfile, cs=1):
        genes = []

        file = pd.read_csv(self.genefile, sep="\t", header=None)

        import json
        with open(resultsfile, "r") as _file:
            gene2cs = json.load(_file)[0]

        coregenes = set([key for key, value in gene2cs.items() if value >= cs])

        for i, row in file.iterrows():
            if row[3] in coregenes:
                genes.append(CoreGene(int(row[0][3:]), row[1], row[2], row[3], gene2cs[row[3]]))

        print("number of core genes: {}".format(len(genes)))

        self.coregenes = GenomicRangeContainer(genes)

    def build_mendelians(self, mendelians: Iterable):
        genes = []

        if not isinstance(mendelians, set):
            mendelians = set(mendelians)

        file = pd.read_csv(self.genefile, sep="\t", header=None)

        for i, row in file.iterrows():
            if row[3] in mendelians:
                genes.append(CoreGene(int(row[0][3:]), row[1], row[2], row[3], 12))

        print("number of mendelian genes: {}".format(len(genes)))

        self.mendelians = GenomicRangeContainer(genes)

    def build_snps(self):
        snps = []

        if self.snp_is_bed:
            file = pd.read_csv(self.snpfile, sep="\t", header=None)

            for i, row in file.iterrows():
                snps.append(SNP(int(row[0][3:]), row[1], row[3]))

        else:
            snp_hgncs = set(pd.read_csv(self.snpfile, header=0, sep=",")["HGNC"].tolist())
            file = pd.read_csv(self.genefile, sep="\t", header=None)

            for i, row in file.iterrows():
                if row[3] in snp_hgncs:
                    snps.append(SNP(int(row[0][3:]), row[1], row[3], end=row[2]))

        print("number of SNPs: {}".format(len(snps)))

        self.snps = GenomicRangeContainer(snps)

    def assign_genes_to_ld_block(self):
        self.ldblocks.integrate(self.genes)

    def assign_snps_to_ld_block(self, offset=0):
        self.ldblocks.integrate(self.snps)
        if hasattr(self, "genes"):
            self.snps.integrate(self.genes, offset=0)

    def assign_coregenes_to_ld_block(self):
        self.ldblocks.integrate(self.coregenes)

    def assign_mendelians_to_ld_block(self):
        self.ldblocks.integrate(self.mendelians)

    def assemble(self, resultsfile=None):
        self.build_ldblocks()
        self.build_genes()
        self.build_snps()

        if resultsfile is not None:
            self.build_coregenes(resultsfile=resultsfile)

        self.assign_genes_to_ld_block()
        self.assign_snps_to_ld_block()

        if resultsfile is not None:
            self.assign_coregenes_to_ld_block()

    def count_ldblocks(self, normalize=False, cs=1):
        import numpy as np
        import pandas as pd

        SNP_and_coregene = 0
        SNP_and_not_coregene = 0
        not_SNP_and_coregene = 0
        not_SNP_and_not_coregene = 0

        rows = []

        for blocks in self.ldblocks.by_chromosome.values():
            for block in blocks:
                num_genes = 0
                has_SNP = False
                has_coregene = False
                genes = []
                snps = []
                coregenes = []
                selected_coregenes = []
                mendelians = []
                for element in block.get_payload():
                    if isinstance(element, CoreGene):
                        if element.cs < 12:
                            coregenes.append(element)
                        if element.cs >= cs and element.cs < 12:
                            has_coregene = True
                            selected_coregenes.append(element)
                        elif element.cs == 12:
                            has_coregene = True
                            mendelians.append(element)
                    elif type(element) is Gene:
                        num_genes += 1
                        genes.append(element)
                    elif isinstance(element, SNP):
                        has_SNP = True
                        snps.append(element)
                
                if has_coregene or has_SNP:
                    rows.append([block,
                                 ", ".join(map(str, snps)),
                                 ", ".join(map(str, mendelians)),
                                 ", ".join(map(str, selected_coregenes)),
                                 ", ".join(map(str, coregenes)),
                                 ", ".join(map(str, genes))])

                weight = 1 if not normalize or num_genes == 0 else 1/num_genes

                if has_SNP:
                    if has_coregene:
                        SNP_and_coregene += weight
                    else:
                        SNP_and_not_coregene += weight
                else:
                    if has_coregene:
                        not_SNP_and_coregene += weight
                    else:
                        not_SNP_and_not_coregene += weight

        array = np.asarray([[SNP_and_coregene, SNP_and_not_coregene],
                           [not_SNP_and_coregene, not_SNP_and_not_coregene]])
        
        if normalize:
            unnormalized_sum = self.count_ldblocks(normalize=False, cs=cs)[0].sum()
            normalized_sum = array.sum()

            array *= unnormalized_sum / normalized_sum

            assert array.sum() == unnormalized_sum

        df = pd.DataFrame(data=rows, columns=["Block", "HSPs", "Mendelian", "CS=11", "CS>=1", "BlockGenes"])

        return array, df

    def check_overlap(self, *args, **kwargs):
        from scipy.stats import fisher_exact
        self.assemble(*args, **kwargs)

        array = self.count_ldblocks()
        return fisher_exact(array)


class GenomicRange:
    def __init__(self, chromosome: int, start: int, end: int, payload=[]):
        self.chromosome = int(chromosome)
        self.start = int(start)
        self.end = int(end)
        self.current = int(start)
        self.payload = payload[:]

    def contains(self, chromosome, position, end=None):
        if chromosome == self.chromosome:
            if end is None:
                return self.start <= position <= self.end
            else:
                return self.start <= position and self.end >= end
        else:
            return False

    def contains_grange(self, grange: 'GenomicRange', offset=0):
        start = self.start - offset
        end = self.end + offset
        if grange.chromosome == self.chromosome:
            #       overlap at end                        or overlap at beginning                 or be completelely contained by grange
            return start <= grange.start <= end or start <= grange.end <= end or (grange.start <= start and grange.end >= end)
        else:
            return False

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current <= self.end:
            return self.current
        raise StopIteration

    def __str__(self):
        return "{}:{}-{}".format(self.chromosome, self.start, self.end)

    def add(self, object):
        self.payload.append(object)

    def carries(self, object):
        return object in self.payload

    def get_payload(self):
        return self.payload
    

class Gene(GenomicRange):
    def __init__(self, chromosome: int, start: int, end: int, name: str):
        super().__init__(chromosome, start, end)
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.name)


class CoreGene(Gene):
    def __init__(self, chromosome: int, start: int, end: int, name: str, cs: int):
        super().__init__(chromosome, start, end, name)
        self.cs = cs


class SNP(Gene):
    def __init__(self, chromosome: int, start: int, name: str, end=None):
        super().__init__(chromosome, start, start if end is None else end, name)

    def __str__(self):
        return "{}({})".format(self.name, ",".join(map(str, self.get_payload())))

    def __repr__(self):
        return "{}({})".format(self.name, ",".join(map(str, self.get_payload())))

class GenomicRangeContainer:
    def __init__(self, ranges=None):
        self.by_chromosome = {}
        identifiers = []
        if ranges is not None:
            for grange in ranges:
                try:
                    self.by_chromosome[grange.chromosome].append(grange)
                except KeyError:
                    self.by_chromosome[grange.chromosome] = [grange]
                identifiers.append(str(grange))
        self.identifiers = set(identifiers)
        self.sort()

    def __repr__(self):
        return self.identifiers

    def intersection(self, other: Union[set, 'GenomicRangeContainer']) -> set:
        if isinstance(other, set):
            return self.identifiers.intersection(other)
        elif isinstance(other, GenomicRangeContainer):
            return self.identifiers.intersection(other.identifiers)
        else:
            raise TypeError("intersection only implemented for sets and GenomicRangeContainers")
        
    def difference(self, other: Union[set, 'GenomicRangeContainer']) -> set:
        if isinstance(other, set):
            return self.identifiers.difference(other)
        elif isinstance(other, GenomicRangeContainer):
            return self.identifiers.difference(other.identifiers)
        else:
            raise TypeError("difference only implemented for sets and GenomicRangeContainers")
        
    def union(self, other: Union[set, 'GenomicRangeContainer']) -> set:
        if isinstance(other, set):
            return self.identifiers.union(other)
        elif isinstance(other, GenomicRangeContainer):
            return self.identifiers.union(other.identifiers)
        else:
            raise TypeError("union only implemented for sets and GenomicRangeContainers")

    def sort(self):
        keys = self.by_chromosome.keys()

        for key in keys:
            self.by_chromosome[key] = sorted(self.by_chromosome[key], key=lambda grange: grange.start)

    def get_by_chromosome(self, chromosome):
        return self.by_chromosome[chromosome]
    
    def integrate(self, other: 'GenomicRangeContainer', offset=0):
        chromosomes = self.by_chromosome.keys()

        for chromosome in chromosomes:
            containing_ranges = self.get_by_chromosome(chromosome)
            try:
                added_ranges = other.get_by_chromosome(chromosome)
            except KeyError:
                continue

            i = 0
            j = 0

            while i < len(containing_ranges) and j < len(added_ranges):
                added_range = added_ranges[j]
                containing_range = containing_ranges[i]

                if containing_range.contains_grange(added_range, offset=offset):
                    containing_range.add(added_range)

                    try:
                        if added_range.end > containing_range.end and added_range.end < containing_ranges[i+1].start:
                            j += 1
                        else:
                            i += 1
                    except IndexError:
                        j += 1

                elif added_range.end < containing_range.start:
                    j += 1
                    try:
                        if i > 0 and added_ranges[j].start < containing_ranges[i-1].end:
                            i -= 1
                    except IndexError:
                        pass

                elif added_range.start > containing_range.end:
                    i += 1

    def __len__(self):
        return len(self.identifiers)
