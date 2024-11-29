#### CREATE CONVERGENCE PLOT 
library(ggplot2)
library(dplyr)
library(tidyr)

files <- list.files(path = ".", pattern = "genetic_algorithm_results_run_.*\\.csv", full.names = TRUE)

read_data <- function(file) {
  df <- read.csv(file, header = TRUE, col.names = c("Genes", "Score", "Length"))
  df$Generation <- 1:nrow(df)  # Crear columna de generación
  df
}

all_data <- lapply(files, read_data)
all_data <- bind_rows(all_data, .id = "Run")

all_data$Score <- as.numeric(all_data$Score)
all_data$Length <- as.numeric(all_data$Length)

p1 <- ggplot(all_data, aes(x = Generation, y = Score, color = Run)) +
  geom_line() +
  geom_point() +  # Agregar puntos a la línea
  #geom_hline(yintercept = 0.662, linetype = "dashed", color = "black") +  # Línea horizontal
  #(aes(x = Generation, y = 0.662), shape = 21, fill = "black", size = 2) +  # Puntos en la línea
  labs(title = "Convergence Plot", x = "Generations", y = "NMI") +
  scale_x_continuous(breaks = seq(1, max(all_data$Generation), by = 1)) +  # Solo números enteros en el eje X
  theme_minimal() 

p2 <- ggplot(all_data, aes(x = Generation, y = Length, color = Run)) +
  geom_line() +
  geom_point() +
  labs(title = "Number of Genes Included", x = "Generations", y = "Length") +
  scale_x_continuous(breaks = seq(1, max(all_data$Generation), by = 1)) +  # Solo números enteros en el eje X
  theme_minimal() +
  theme(legend.position = "none")

# Combinar ambos gráficos
library(gridExtra)
combined_plot <- grid.arrange(p1, p2, ncol = 2)
ggsave("Seurat_solito.png", plot = combined_plot, width = 8, height = 4, dpi = 72)
grid.arrange(p1, p2, ncol = 2)


###### SIGNIFICANCIA BIOLOGICA ####


# Extract genes from the 'Individual' column
genes_list <- strsplit(as.character(all_data$Genes), " ")

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
