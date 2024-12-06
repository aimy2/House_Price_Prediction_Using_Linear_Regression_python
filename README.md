# House_Price_Prediction_Using_Linear_Regression_python

## Table of Content
1. [Description](#Description)
2. [Problem Statement](#ProblemStatement)
3. [Features](#Features)
4. [Technologies Used](#TechnologiesUsed)
5. [Intallation](#Installation)
6. [Usage](#Usage)
7. [Contact](#Contact)
   


### Description
This model predicts the any house price based on the features provided as it did supervised learning.

### Problem Statement
Houses prices are not predictable accurately.

### Features
- Regression model
- Real-time price predictions on housing related features.

### Technologies Used

- **Python**
- **Numpy**
- **Sklearn**
- **Seaborn**
- **Pandas**
- **Matpltlib**
- **TensorBoard**

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/medical-image-classification.git
   cd medical-image-classification 
2. Install dependencies:
   pip install -r requirements.txt
3. Run the Flask app (if applicable):
   python app.py  

### Usage
How to run the project once it’s set up. This could include training the model, making predictions, and interacting with any interface
1. **Training the Model:**
      To train the model, run:
       python
      python train.py 
2. **Making Predictions**: After training, use the following code to make predictions on new medical images:
      ```import pandas as pd
      from sklearn.linear_model import LinearRegression
      from sklearn.model_selection import train_test_split
      from sklearn.metrics import mean_squared_error
      
      # Load dataset
      data = pd.read_csv('house_data.csv')  # Make sure the CSV file is in the right directory
      
      # Preprocess data (select relevant features)
      X = data[['area', 'bedrooms', 'bathrooms']]  # Example features
      y = data['price']
      
      # Split the data into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      
      # Train the model
      model = LinearRegression()
      model.fit(X_train, y_train)
      
      # Make predictions
      predictions = model.predict(X_test)
      
      # Evaluate the model
      mse = mean_squared_error(y_test, predictions)
      print(f"Mean Squared Error: {mse}")
      
      # Example prediction for a new house
      new_house = pd.DataFrame([[2500, 4, 3]], columns=['area', 'bedrooms', 'bathrooms'])
      predicted_price = model.predict(new_house)
      print(f"Predicted price for the new house: ${predicted_price[0]:,.2f}")```

### Contact
- [Syeda Aiman Mumtaz Sherazi](mailto:aimanmumtaz27@gmail.com)
  
### just checking
![different colour](https://github.com/user-attachments/assets/80100bbf-cb50-4f04-aedf-b22856821e17)
