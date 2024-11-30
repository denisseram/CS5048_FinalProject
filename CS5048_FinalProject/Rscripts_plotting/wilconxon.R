#wilcoxon_rank for the marker genes or seurat

library(dplyr)

max_nmi_per_run <- all_data %>%
  group_by(Run) %>%
  filter(Score == max(Score)) %>%
  slice(1) %>%  # Elegir la primera línea en caso de empates
  ungroup()

# Ver los resultados
head(max_nmi_per_run)

max_nmi_per_run_seurat <- all_data_seurat %>%
  group_by(Run) %>%
  filter(Score == max(Score)) %>%
  slice(1) %>%  # Elegir la primera línea en caso de empates
  ungroup()

# Ver los resultados
head(max_nmi_per_run_seurat)

scores_1 <- max_nmi_per_run %>% pull(Score)
scores_2 <- max_nmi_per_run_seurat %>% pull(Score)

wilcox_test <- wilcox.test(scores_1, scores_2, alternative = "two.sided")

wilcox_test

library(tibble)
library(ggplot2)

# Crear un dataframe combinado para graficar
data_for_plot <- bind_rows(
  tibble(Group = "with marker genes", Score = scores_1),
  tibble(Group = "without marker genes", Score = scores_2)
)

# Generar el box plot
ggplot(data_for_plot, aes(x = Group, y = Score, fill = Group)) +
  geom_boxplot() +
  labs(
    title = "Comparison of NMI values between implementation with marker genes and without marker genes",
    x = "Grupo",
    y = "Score"
  ) +
  theme_minimal() +
  theme(legend.position = "none")