# Load required libraries
library(Seurat)              # For single-cell RNA-seq data analysis
library(SingleCellExperiment) # To handle single-cell data objects
library(dplyr)               # For data manipulation
library(patchwork)           # For combining plots
require(graphics)            # For basic graphics
library(graphics)            # For additional graphics support
library(ggplot2)             # For data visualization

# Function to filter marker genes from a dataset
filter_markerGenes <- function(filename_dataset){
  # Load the SingleCellExperiment (SCE) object
  sce <- readRDS(paste0("./CS5048_FinalProject/SOURCE/", filename_dataset, ".RDS"))
  
  # Read the list of marker genes from a text file
  allmarkers <- read.table("./filter_genes.txt")
  
  # Filter the SCE object to retain only genes
  filtered_sce <- sce[rownames(sce) %in% allmarkers$V1, ]
  
  # Save the filtered SCE object
  saveRDS(filtered_sce, file = paste0("./CS5048_FinalProject/SOURCE/", filename_dataset, "_markergenes.rds"))
}

# Function to perform clustering using Seurat
seurat_clustering <- function(filename_dataset){
  # Extract dataset name for folder structure
  dataset = paste0(strsplit(filename_dataset, "_")[[1]][1], "_", strsplit(filename_dataset, "_")[[1]][2])
  folder = paste0("./CS5048_FinalProject/results/", dataset, "/seurat")

  # Create output folder if it doesn't exist
  if (!file.exists(folder)) {
    dir.create(folder, recursive = TRUE)
  }
  
  # Load the filtered dataset
  sce <- readRDS(paste0("./CS5048_FinalProject/SOURCE/", filename_dataset, ".RDS"))

  # Create a Seurat object from the SCE object
  seu <- CreateSeuratObject(
    counts = counts(sce), 
    min.cells = 2, 
    project = "test", 
    min.features = 0, 
    names.field = 1
  )
  
  # Normalize the data using log-normalization
  seu <- NormalizeData(seu, normalization.method = "LogNormalize", scale.factor = 10000, display.progress = FALSE)
  
  # Identify highly variable features
  all.genes <- rownames(seu)  # Use all genes for scaling

  
  print(dim(counts(sce)))
  seu <- ScaleData(seu, features = all.genes)
  
  # Perform PCA for dimensionality reduction
  set.seed(1234567) # Ensure reproducibility
  seu <- RunPCA(seu, features = all.genes)  # Use all genes for PCA
  
  # Visualize PCA results (optional Elbow plot to determine optimal dimensions)
  ElbowPlot(seu)
  
  # Perform clustering
  seu <- FindNeighbors(seu, dims = 1:10)
  seu <- FindClusters(seu, resolution = 0.5)
  
  # Perform non-linear dimensionality reduction using UMAP
  seu <- RunUMAP(seu, dims = 1:10)

  
  # Add metadata to Seurat object
  seu <- AddMetaData(seu, metadata = seu@meta.data$RNA_snn_res.0.5, col.name = "seurat_cluster")
  seu <- AddMetaData(seu, metadata = colData(sce)$cellType, col.name = "cellType")
  
  # Generate and save UMAP plot
  umap <- DimPlot(
    seu, reduction = "umap", label = FALSE, label.size = 4.5, 
    group.by = c("cellType", "seurat_cluster")
  ) + xlab("UMAP 1") + ylab("UMAP 2") +
    theme(axis.title = element_text(size = 18), legend.text = element_text(size = 18)) +
    guides(colour = guide_legend(override.aes = list(size = 10)))

  ggsave(
  filename = paste0(folder, "umap_seurat_", filename_dataset, ".png"),
  plot = umap
    )  
  
  
  # Save cluster assignments to CSV
  res <- seu@meta.data$RNA_snn_res.0.5
  final <- data.frame(colnames(seu), as.numeric(res))
  colnames(final) <- c("cell", "seurat")
  write.csv(final, file = paste0(folder, "/seurat.csv"), sep = ",", quote = FALSE, row.names = TRUE, col.names = TRUE)
  
 return(dim(counts(sce)))
}

# Function to calculate Normalized Mutual Information (NMI)
calc_NMI <- function(M, N) {
  library("aricode")
  if (length(M) != length(N)) {
    print("Error. The length of vectors should be the same.")
    NMI <- NA
  } else {
    NMI <- NMI(M, N)
  }
  return(NMI)
}

calc_ARI <- function(M, N) {
  
  library("mclust")
  
  if (length(M) != length(N)) {
    print("Error. The length of vectors should be the same.")
    ARI <- NA
  } else {
    ARI <- adjustedRandIndex(M,N)
  }
  return(ARI)
}

# workflow execution
filter_markerGenes("muraro") # Filter marker genes for the "muraro" dataset
min <- seurat_clustering("muraro_markergenes") # Perform Seurat clustering on filtered data
print("DIMENSIONES")
print(min)
# Load results and calculate NMI
dataset <- "muraro_markergenes"
sce <- readRDS(paste0("./CS5048_FinalProject/SOURCE/", dataset, ".RDS"))
resultados <- read.csv(paste0("./CS5048_FinalProject/results/", dataset, "/seurat/seurat.csv"))

# Retrieve cell type annotations and cluster assignments
cell_annotations <- colData(sce)
cell_type_annotations <- cell_annotations$cellType
cluster_assignation <- resultados[["seurat"]]

# Calculate NMI score
nmi <- calc_NMI(cell_type_annotations, cluster_assignation)

# Determine the number of unique clusters
cluster_assignation1 <- as.data.frame(cluster_assignation)
num_clu <- length(unique(cluster_assignation1$cluster_assignation))

print(nmi)
print(num_clu)

# Output NMI score
score <- as.numeric(nmi)
print(score)

cat(score)
