#!/usr/bin/Rscript
msg.out <- capture.output(suppressMessages(library(M3Drop)))

args <- commandArgs(trailingOnly=TRUE)
fname = args[1]
transpose = as.logical(args[2])

df <- read.csv(fname, sep='\t', header=TRUE, row.names=1)
if(!transpose){
	df <- t(df)
}

DE_genes <- M3DropDifferentialExpression(df, mt_method="fdr", mt_threshold=0.01)

write.table(round(t(df[as.character(DE_genes$Gene), ]), digits=5), file=stdout(), sep='\t', quote=FALSE)
