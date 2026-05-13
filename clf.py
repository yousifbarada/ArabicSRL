import pickle

import xgboost as xgb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import label encoder
from sklearn.preprocessing import LabelEncoder
import torch 
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
label_encoder = LabelEncoder()
data_dict = pickle.load(open('./data.pkl', 'rb'))

data, labels = data_dict
print(len(data))
print(set(len(x) for x in data))
data = np.array(data)
print(len(set(labels)))
labels = label_encoder.fit_transform(np.array(labels))
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# =========================
# DataLoader
# =========================

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=32,
    shuffle=True
)

# =========================
# Model
# =========================

model = nn.Sequential(
    nn.Linear(42, 128),
    nn.ReLU(),

    nn.Linear(128, 64),
    nn.ReLU(),

    nn.Linear(64, 28)
)

# =========================
# Loss + Optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================
# Training loop
# =========================

for epoch in range(40):

    model.train()

    total_loss = 0

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =========================
# Evaluation
# =========================

model.eval()

with torch.no_grad():

    outputs = model(x_test)

    predictions = torch.argmax(outputs, dim=1)

    accuracy = (predictions == y_test).float().mean()

print(f"Accuracy: {accuracy*100:.2f}%")

# =========================
# Save model
# =========================

torch.save(model.state_dict(), "model.pth")

print("Model saved!")
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

# model = xgb.XGBClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.pkl', 'wb')
# pickle.dump({'model': model}, f)
# f.close()