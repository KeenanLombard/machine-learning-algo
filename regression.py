import numpy as np

# Logistic regression parameters
intercepts = [-2.8690, 0.1485, -19.3199]  # Coefficients for intercept, rebounds, and points features
std_errors = [0.0161, 0.0299]  # Standard errors for rebounds and points features
z_statistics = [4.3235, 5.6734]  # Z-statistics for rebounds and points features
p_values = ['<0.0001', '<0.0001']  # P-values for rebounds and points features

# Function to calculate probability of default
def calculate_probability(rebounds, points):
    # Calculate log odds
    log_odds = intercepts[0] + intercepts[1] * rebounds + intercepts[2] * points
    
    # Calculate probability using logistic function
    probability = 1 / (1 + np.exp(-log_odds))
    
    return probability

# Sample data (rebounds and points for each player)
sample_data = [
    {"rebounds": 8, "points": 15},
    {"rebounds": 3, "points": 7}
]

# Predict probabilities for each player
for i, player in enumerate(sample_data):
    probability = calculate_probability(player["rebounds"], player["points"])
    
    print(f"Player {i+1}:")
    print(f"Rebounds: {player['rebounds']}")
    print(f"Points: {player['points']}")
    print(f"Probability of being drafted: {probability:.3f}")
    print()
