# NFL-Season-Simulation


Write-up is coming soon. ETA: First week of September

In the meantime, here are some results of my 2024 win totals compared to popular sportsbooks by RMSE:


Mine: 3.11
Caesars: 3.23
Draft Kings: 3.27
ESPN Bet: 3.34


## Basic Overview

Created a game model that predicts the probability of a home team win using:
- home/away epa
- home/away ppg
- home/away ppga
- home/away wins

I then generated forecasts for the 2025 season using a Bayesian framework, drawing from the posterior predictive distribution of the model parameters. This allowed me to incorporate posterior uncertainty into the forecasts, capturing the inherent variability in NFL game outcomes from week to week.
