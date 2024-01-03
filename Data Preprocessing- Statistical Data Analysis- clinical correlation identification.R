library(readxl)
library(MetaboAnalystR)
library(ComplexHeatmap)
library(circlize)
library(dplyr)
library(ggplot2)

cytokines <- read_excel('Data.xlsx', sheet = 'Cytokines')
Proteins <- read_excel('Data.xlsx', sheet = 'Proteins')
Metabolites <- read_excel('Data.xlsx', sheet = 'Metabolites')

Data <- cbind(cytokines, 
              Proteins[5:length(Proteins)], 
              Metabolites[5:length(Metabolites)])

Data <- cbind(Data[,1:4], 
              as.data.frame(sapply(Data[,5:length(Data)],
                                               function (x) as.numeric(as.character(x)))))

sum(Data == 0, na.rm = TRUE)
Data <- replace(Data, Data == 0, NA)
sum(Data == 0, na.rm = TRUE) #Should return 0

Transposed_Data <- as.data.frame(t(Data))
## missing value vector:
p <-c()
for (i in 1:nrow(Transposed_Data)) {
  p[i] <- sum(is.na(Transposed_Data[i,]))/262
 } 
#input the vector to the data
Transposed_Data <- Transposed_Data %>% 
  mutate(percent_of_missing_Vlues= p) %>% 
  select(percent_of_missing_Vlues, everything())
#filter the data based on the generated column (percent of #missing values)
filtered_Transposed_Data <- Transposed_Data %>% 
  filter(percent_of_missing_Vlues < 0.5)
##re-transpose the data:
cleaned_data <- as.data.frame(t(filtered_Transposed_Data))
cleaned_data <- cleaned_data[-1,]
row.names(cleaned_data) <- row.names(data)

## imputation of remaining missing values
sum(is.na(cleaned_data))
for (i in 5:782){
  for(j in 1:262){
    if (is.na(cleaned_data[j,i]) == "TRUE"){
      cleaned_data[j,i] = min(cleaned_data[,i], na.rm = TRUE)
    }
  }
  }

##check NAs and zero values to be removed
sum(is.na(cleaned_data))
sum(cleaned_data[,5:782]== 0, na.rm = TRUE)

mSet<-InitDataObjects("conc", "stat", FALSE)
mSet<-Read.TextData(mSet, "cleaned data.csv", "rowu", "disc")
mSet<-SanityCheckData(mSet)

mSet<-PreparePrenormData(mSet)
mSet<-Normalization(mSet, "NULL", "LogNorm","MeanCenter", "NULL", ratio=FALSE, ratioNum=20)   

mSet<-PCA.Anal(mSet) #Perform PCA
mSet<-PlotPCAPairSummary(mSet, "pca_pair_0_", format = "png", dpi = 72, width=NA, 5) # Create PCA overview
mSet<-PlotPCAScree(mSet, "pca_scree_0_", "png", dpi = 72, width=NA, 5) # Create PCA scree plot
mSet<-PlotPCA2DScore(mSet, "pca_score2d_0_", format = "png", dpi=300, width=NA, 1, 2, 0.95, 0, 0) # Create a 2D PCA score plot

mSet<-Volcano.Anal(mSet, FALSE, 1.5, 0, F, 0.05, TRUE, "fdr") 

mSet<-PlotVolcano(mSet, "volcano_0_", 1, 0, format ="png",    dpi=300, width=NA)

mSet<-PlotSubHeatMap(mSet, "heatmap_1_", "png", 300, width=NA,"norm", "row", "euclidean", "ward.D","bwm", 8,
                     "tanova", 100, "overview", F, T, T, F, T, T, T)

clinical_variables<-read.csv("Path_to_your_data.csv")

Coefficients<-data.frame(matrix(nrow = n_o_clinical variables ,ncol = n_of_differentially_changed_molecules )) #empty data frame for odds ratios
row.names(Coefficients)<-colnames(clinical_variables [220:239])
colnames(Coefficients)<-colnames(clinical_variables [1:219])

pvalue<-data.frame(matrix(nrow = n_o_clinical variables, ncol = n_of_differentially_changed_molecules)) #empty data frame for p values
row.names(pvalue)<-colnames(clinical_variables [220:239])
colnames(pvalue)<-colnames(clinical_variables [1:219])

for(i in 1:#where molecules finished){ 
     for(j in #where clinical variables start:where it finished){ 
         sym<-glm(unlist(clinical_variables [j])~unlist(clinical_variables [i])+
                  unlist(clinical_variables [244])+ 
                  unlist(clinical_variables [245])+
                  unlist(clinical_variables [246])+ 
                  unlist(clinical_variables [247])+
                  unlist(clinical_variables [248])+
                  unlist(clinical_variables [249])+
                  unlist(clinical_variables [250])+
                  unlist(clinical_variables [251])+
                  unlist(clinical_variables [252]),family=binomial(),
                  data= clinical_variables)
         pvalue [j-219,i]<-coef(summary(sym))[2,4]
         Coefficients [j-219,i]<-exp(coef(summary(sym))[2,1])
         }  
} 

# transposing the matrix 
n.pval <- as.data.frame(t(pvalue)) 
#Make a column for the number of significant p values in each row for transposed table
for (i in 1:nrow(n.pval)){ 
  n.pval[i,21]<- sum(n.pval[i,] < 0.05)
}
# subset of molecules with >= 3 significant p values
n.pval.filtered<- n.pval %>% 
  filter(n.of.sig >= 3)
#subset of odds ratio values of molecules with >= 3 significant p values
odds.filrtered <- new.oddsratio %>% 
  select(rownames(n.pval.filtered))

Heatmap(
  as.matrix(odds.filrtered),
  width = ncol(odds.filrtered)*unit(4.5, "mm"), 
  height = nrow(odds.filrtered)*unit(3, "mm"),
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  show_row_dend = FALSE,
  show_column_dend = FALSE,
  clustering_method_rows = "ward.D2",
  heatmap_legend_param = list(
    title = "Odds Ratio",
    at = seq(0, 2, length.out = 3),
    labels = c( "10^-5", "1", "10^5")),
  cell_fun = function(j, i, x, y, w, h, fill) {
    if(n.pval.filtered [i, j] < 0.05) {
      grid.text("*", x , y-(0.3*h) ,gp=gpar(fontsize=18))
    }
  },
  col = my_color_mapping,
  column_names_gp = grid::gpar(fontfamily= "Arial",fontface= "bold" ,fontsize = 10),
  row_names_gp = grid::gpar(fontfamily= "Arial",fontface= "bold",fontsize = 9)
)










