# How to Guide 
1. Create a folder on the desktop
2. Open the folder in Visual Studio Code
3. Create a file named <b>weightmodel.ipynb</b>
4. Add a code cell
5. [Navigate to the following URL on Google](https://www.kaggle.com/datasets/burnoutminer/heights-and-weights-dataset)
6. Select the download button
7. Copy the code and paste it into the code cell, in Visual Studio Code
8. Open a new terminal in Visual Studio Code
9. Install Python by running the following command: ```python -m pip install numpy pandas tensorflow scikit-learn kagglehub```
10. Check if the latest version has been installed, by running the following command: ```python --version```
11. Ensure the path to dataset files are displayed.
12. Paste the following code in a new code cell:
```
for file in os.listdir(path):
    print(file)
```
13. Run the code by selecting the Execute arrow button on the left of the code cell.
14. Paste the following code in a new code cell:
```
df=pd.read_csv(os.path.join(path, "SOCR-HeightWeight.csv" ))
print(df.head())
print(df.columns)
```
15. Run the code by selecting the Execute arrow button on the left of the code cell.
16. Paste the following code in a new code cell:
```
df["Height_cm"] = df["Height(Inches)"] * 2.54
df["Weight_kg"] = df["Weight(Pounds)"] * 0.45359237
```
17. Run the code by selecting the Execute arrow button on the left of the code cell.
18. Paste the following code in a new code cell:
```
data=df[["Height_cm","Weight_kg"]]
print(data.head())
```
19. Paste the following code in a new code cell:
```
x = data[["Height_cm"]].values.reshape(-1,1)
y = data["Weight_kg"].values.reshape(-1,1)
```
20. Run the code by selecting the Execute arrow button on the left of the code cell.
21. Paste the following code in a new code cell:
```
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_scaled=scaler_X.fit_transform(x)
Y_scaled=scaler_Y.fit_transform(y)
```
22. ..
```
base = tf.keras.Sequential([
layers.Input(shape=(1,)),
layers.Dense(1)
])
```
23. 
