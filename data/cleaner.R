library(stringr)
smiles1 <- data.table::fread("bbbp_raw.csv")
filtered_smiles1 <- smiles1[nchar(gsub("[^A-Za-z]", "", smiles1$smiles)) > 10, ]
not1 <- smiles1[nchar(gsub("[^A-Za-z]", "", smiles1$smiles)) == 10, ]
data.table::fwrite(filtered_smiles1,"bbbp.csv")

smiles2 <- data.table::fread("hiv_raw.csv")
filtered_smiles2 <- smiles2[nchar(gsub("[^A-Za-z]", "", smiles2$smiles)) > 10, ]
not2 <- smiles2[nchar(gsub("[^A-Za-z]", "", smiles2$smiles)) == 10, ]
data.table::fwrite(filtered_smiles2,"hiv.csv")

smiles3 <- data.table::fread("tox21_raw.csv")
filtered_smiles3 <- smiles3[nchar(gsub("[^A-Za-z]", "", smiles3$smiles)) > 10, ]
not3 <- smiles3[nchar(gsub("[^A-Za-z]", "", smiles3$smiles)) == 10, ]
data.table::fwrite(filtered_smiles3,"tox21.csv")

sum(filtered_smiles2$label)
sum(smiles2$label)
