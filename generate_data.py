import pandas as pd
import random

# Define crops and their typical environmental conditions
crops = [
    'Rice', 'Wheat', 'Corn', 'Cotton', 'Barley', 'Tomato', 'Potato', 'Carrot', 
    'Cucumber', 'Soybean', 'Pumpkin', 'Sorghum', 'Millet', 'Banana', 'Groundnut', 
    'Peas', 'Onion', 'Ginger', 'Spinach', 'Cabbage', 'Brinjal', 'Lettuce', 'Chili',
    'Garlic', 'Papaya', 'Pineapple', 'Radish', 'Cauliflower', 'Guava', 'Watermelon',
    'Mango', 'Orange', 'Sugarcane', 'Beetroot', 'Strawberry', 'Coffee', 'Tea', 'Rubber', 
    'Peach', 'Grapes', 'Blackberry', 'Lemon', 'Lime', 'Cashew', 'Cocoa', 'Apple', 'Blueberry',
    'Peanut', 'Zucchini', 'Avocado', 'Cantaloupe', 'Broccoli', 'Plum', 'Cherry'
]

# Generate random data
data = []
for i in range(1000):
    temp = random.uniform(15, 40)      # Temperature in degrees Celsius
    humidity = random.uniform(50, 90)  # Humidity in percentage
    ph = random.uniform(5.5, 7.5)      # pH of soil
    moisture = random.uniform(15, 40)  # Soil moisture in percentage
    rainfall = random.uniform(100, 400) # Rainfall in mm
    crop = random.choice(crops)        # Random crop from the list
    
    data.append([temp, humidity, ph, moisture, rainfall, crop])

# Create a DataFrame
df = pd.DataFrame(data, columns=['temperature', 'humidity', 'ph', 'moisture', 'rainfall', 'label'])

# Save to CSV
df.to_csv('crop_recommendation.csv', index=False)

print("Data generated and saved as 'crop_recommendation.csv'")
