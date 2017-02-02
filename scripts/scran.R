#!/usr/bin/Rscript
msg.out <- capture.output(suppressMessages(library(scran)))

args <- commandArgs(trailingOnly=TRUE)
fname = args[1]
transpose = as.logical(args[2])
do.cluster = as.logical(args[3])
sizes = as.vector(args[4])
min.size = as.numeric(args[5])
positive = as.logical(args[5])

df <- read.csv(fname, sep='\t', header=TRUE, row.names=1)

if (!transpose) {
	df <- t(df)
}

sce <- newSCESet(countData=df)

if (do.cluster) {
	qclust <- quickCluster(sce, min.size=min.size)
	sce <- computeSumFactors(sce, clusters=qclust, positive=positive)
} else {
	sce <- computeSumFactors(sce, positive=positive)
}

sce <- normalize(sce)
write.table(round(t(exprs(sce)), digits=5), file=stdout(), sep='\t', quote=FALSE)
