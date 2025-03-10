# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

![Screenshot 2025-03-03 113429](https://github.com/user-attachments/assets/a9209adf-ffbf-42c7-8883-d8d57ab1df75)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:KRITHIGA U
### Register Number:212223240076
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8) # First linear layer with input size 1 and output size 8
        self.fc2=nn.Linear(8,4) # Second linear layer with input size 8 and output size 4
        self.fc3=nn.Linear(4,1) # Third linear layer with input size 4 and output size 1
        self.relu=nn.ReLU()
        self.history={'loss':[]}
  def forward(self,x):
        x=self.relu(self.fc1(x)) # Pass input through fc1 and apply ReLU activation
        x=self.relu(self.fc2(x)) # Pass output of fc1 through fc2 and apply ReLU activation
        x=self.fc3(x)            # Pass output of fc2 through fc3
        return x
ai_brain= NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)



# Initialize the Model, Loss Function, and Optimizer
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
## Dataset Information

![Screenshot 2025-03-02 225630](https://github.com/user-attachments/assets/9fbf696e-824e-411e-a79a-2123af1862e6)



## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2025-03-03 112603](https://github.com/user-attachments/assets/85c4707d-6059-4489-9d28-e14e7e81f1c1)



### New Sample Data Prediction

![Screenshot 2025-03-03 112651](https://github.com/user-attachments/assets/1e6cb934-939b-4d9b-aacf-d6dd548010d3)

## RESULT

Thus a neural network regression model for the dataset was developed successfully.
