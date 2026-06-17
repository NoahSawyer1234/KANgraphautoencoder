library(stringr)
smiles <- data.table::fread("chembl_small_organic.csv")
filtered_smiles <- smiles[nchar(gsub("[^A-Za-z]", "", smiles$canonical_smiles))>10,]
filtered_smiles$tag=1
molecules = data.frame(smiles=filtered_smiles$canonical_smiles,label=filtered_smiles$tag)
molecules <- molecules[sample(nrow(molecules)), ]

bace <- data.table::fread("data/bace.csv")
bace_smiles <- bace$smiles
bace_in_main <- bace_smiles %in% molecules$smiles
sum(bace_in_main)

hiv <- data.table::fread("data/hiv.csv")
hiv_smiles <- hiv$smiles
hiv_in_main <- hiv_smiles %in% molecules$smiles
sum(hiv_in_main)

bbbp <- data.table::fread("data/BBBP.csv")
bbbp_smiles <- bbbp$smiles
bbbp_in_main <- bbbp_smiles %in% molecules$smiles
sum(bbbp_in_main)

tox <- data.table::fread("data/tox21.csv")
tox_smiles <- tox$smiles
tox_in_main <- tox_smiles %in% molecules$smiles
sum(tox_in_main)

no_hiv <- molecules[!molecules$smiles %in% hiv_smiles,]
no_bbbp <- no_hiv[!no_hiv$smiles %in% bbbp_smiles,]
no_tox <- no_bbbp[!no_bbbp$smiles %in% tox_smiles,]
molecules <- no_tox

data.table::fwrite(molecules,"small_molecules.csv")

data.table::fwrite(molecules[1:200000,],"small_molecules_200k.csv")

data.table::fwrite(molecules[1:500000,],"small_molecules_500k.csv")

data.table::fwrite(molecules[1:1000000,],"small_molecules_1m.csv")

