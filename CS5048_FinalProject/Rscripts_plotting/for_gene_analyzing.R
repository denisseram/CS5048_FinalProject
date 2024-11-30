#FOR MAKING ANALYSIS OF GENES
library(ggplot2)
library(dplyr)
library(tidyr)
read_data <- function(file) {
  df <- read.csv(file, header = TRUE, col.names = c("Genes", "Score", "Length"))
  df$Generation <- 1:nrow(df)  # Crear columna de generación
  df
}

all_data <-read_data("/Users/cdr_c/Seurat_solito_genes/genetic_algorithm_results_run_1.csv")
all_data <- bind_rows(all_data, .id = "Run")

all_data$Score <- as.numeric(all_data$Score)
all_data$Length <- as.numeric(all_data$Length)

genes_generacion_5 <- all_data$Genes[all_data$Generation == 5]

genes_list <- strsplit(as.character(genes_generacion_5), " ")

genes_data <- data.frame(Gene = unlist(genes_list), Run = rep(all_data$Run, sapply(genes_list, length)))

gene_freq <- genes_data %>%
  group_by(Gene) %>%
  summarise(Frequency = n()) %>%
  arrange(desc(Frequency))

top_genes <- gene_freq[1:10, ]

ggplot(top_genes, aes(x = reorder(Gene, Frequency), y = Frequency)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Top 10 Most Frequent Genes Across Runs",
       x = "Gene", y = "Frequency")


genes <- unlist(genes_list) # Convierte la lista a un vector plano

# Elimina las comas finales de cada gen, si las hay
genes <- gsub(",$", "", genes)

# Guarda los genes en un archivo de texto, cada gen en una línea
writeLines(genes, "genes_list.txt")

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
  allmarkers <- read.table("./genes_list.txt")
  
  # Filter the SCE object to retain only genes
  filtered_sce <- sce[rownames(sce) %in% allmarkers$V1, ]
  
  # Save the filtered SCE object
  saveRDS(filtered_sce, file = paste0("./CS5048_FinalProject/SOURCE/", filename_dataset, "_markergenes1.rds"))
}

# Function to perform clustering using Seurat
seurat_clustering <- function(filename_dataset){
  filename_dataset <- "muraro_markergenes1"
  # Extract dataset name for folder structure
  dataset = paste0(strsplit(filename_dataset, "_")[[1]][1], "_", strsplit(filename_dataset, "_")[[1]][2])
  folder = paste0("./CS5048_FinalProject/results/", dataset, "/seuratA")
  
  # Create output folder if it doesn't exist
  if (!file.exists(folder)) {
    dir.create(folder, recursive = TRUE)
  }
  
  # Load the filtered dataset
  sce <- readRDS(paste0("./CS5048_FinalProject/SOURCE/", filename_dataset, ".rds"))
  
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
  
  output_path <- file.path(folder, "VlnPlot.png")
  
  vln_plot <- VlnPlot(seu, features = c("RNH1", "PGA5", "MALAT1", "NDN", "MAP3K4", "DHX8"))
  VlnPlot(seu, features = c("RNH1", "PGA5", "MALAT1", "NDN", "MAP3K4", "DHX8"))
  
  
  ggsave(
    filename = output_path,
    plot = vln_plot,
    width = 5.5, # Ancho adecuado para una columna
    height = 4.5, # Altura ajustada según sea necesario
    units = "in", # Unidades en pulgadas
    dpi = 72     # Resolución para impresión
  )
  
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



# workflow execution
filter_markerGenes("muraro") # Filter marker genes for the "muraro" dataset
min <- seurat_clustering("muraro_markergenes1") # Perform Seurat clustering on filtered data
print("DIMENSIONES")
