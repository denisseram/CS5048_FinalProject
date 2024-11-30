#grafico marker genes

library(ggplot2)
library(dplyr)
library(tidyr)

# Leer todos los archivos con un patrón específico
files <- list.files(path = ".", pattern = "genetic_algorithm_results_run_.*\\.csv", full.names = TRUE)

read_data <- function(file) {
  df <- read.csv(file, header = TRUE, col.names = c("Genes", "Score", "Length"))
  df$Generation <- 1:nrow(df)  # Crear columna de generación
  df$LengthCategory <- case_when(
    nrow(df) == 10 ~ "10",
    nrow(df) == 20 ~ "20",
    nrow(df) == 30 ~ "30",
    TRUE ~ "Unknown"
  )
  df
}

# Leer y combinar datos
all_data <- lapply(files, read_data)
all_data <- bind_rows(all_data, .id = "Run")

all_data$Score <- as.numeric(all_data$Score)
all_data$Length <- as.numeric(all_data$Length)

# Gráfico 1: Convergence Plot
p1 <- ggplot(all_data, aes(x = Generation, y = Score, color = Run, linetype = LengthCategory)) +
  geom_line() +
  geom_point() +
  labs(title = "Convergence Plot", x = "Generations", y = "NMI") +
  scale_x_continuous(breaks = seq(1, max(all_data$Generation, na.rm = TRUE), by = 1)) +
  theme_minimal()

# Gráfico 2: Number of Genes Included
p2 <- ggplot(all_data, aes(x = Generation, y = Length, color = Run, linetype = LengthCategory)) +
  geom_line() +
  geom_point() +
  labs(title = "Number of Genes Included", x = "Generations", y = "Length") +
  scale_x_continuous(breaks = seq(1, max(all_data$Generation, na.rm = TRUE), by = 1)) +
  theme_minimal() +
  theme(legend.position = "none")

# Combinar ambos gráficos
library(gridExtra)
combined_plot <- grid.arrange(p1, p2, ncol = 2)
ggsave("Seurat_solito.png", plot = combined_plot, width = 10, height = 5, dpi = 300)
grid.arrange(p1, p2, ncol = 2)
