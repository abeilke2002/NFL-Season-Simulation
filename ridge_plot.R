library(tidyverse)
library(ggridges)
library(nflfastR)

dataset <- read_csv("/Users/aidanbeilke/Desktop/Football Projects/r_simulated.csv") |> 
  select(-wins)

long_df <- dataset %>%
  pivot_longer(cols = starts_with("sim_"),
               names_to = "simulation",
               values_to = "wins")

team_colors = teams_colors_logos

long_df <- long_df %>%
  left_join(teams_colors_logos, by = c("team" = "team_abbr"))

ggplot(long_df, aes(x = wins, y = fct_reorder(team_name, row_mean), fill = team)) +
  geom_density_ridges(
    alpha = 0.95,
    scale = 1.5,
    rel_min_height = 0.01,
    color = "black",
    linewidth = 1
  ) +
  facet_wrap(~ division, scales = "free_y") +
  scale_fill_manual(values = setNames(teams_colors_logos$team_color, teams_colors_logos$team_abbr)) +
  theme_linedraw() +  
  theme(
    panel.grid.major.y = element_line(size = 1),  # Thicker horizontal grid lines
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 12),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.line.x = element_line(size = 1.5, color = "black"),
    axis.text.y = element_text(face = "bold"),
    axis.text.x = element_text(size = 10)
  )
