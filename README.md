# Decision Tree: Complete Guide with Play Tennis Dataset Calculation  

## Section 1: Definition of Decision Tree  

A Decision Tree is a supervised machine learning algorithm used for classification and regression tasks. It splits data into smaller subsets using if-else conditions, forming a tree-like structure.  

### Key Points:  
- Works on both classification and regression tasks.  
- Follows a tree structure with root, branches, and leaves.  
- Uses metrics like Entropy, Gini Index, and Information Gain.  
- Easy to interpret and visualize.  

---

## Section 2: Characteristics of Decision Tree  

- **Hierarchy-Based:** Organizes data in a tree format.  
- **Rule-Based:** Uses conditions to make predictions.  
- **Recursive Splitting:** Splits nodes based on attribute importance.  
- **Overfitting Risk:** Can become too complex with deep trees.  
- **Handles Categorical & Numerical Data:** Works with different data types.  

---

## Section 3: Important Formulas in Decision Tree  

### 1. Entropy Formula  

Entropy measures the impurity of a dataset.  

H(S) = - p1 log2 p1 - p2 log2 p2  

where,  
- p1, p2 = Probability of each class  

### 2. Information Gain (IG) Formula  

Information Gain determines the best attribute for splitting.  

IG = H(Parent) - ∑ ( |Child| / |Parent| * H(Child) )  

### 3. Gini Index Formula  

Measures how often a randomly chosen element is incorrectly classified.  

Gini = 1 - ∑ p(i)²  

where p(i) is the probability of each class.  

---

## Section 4: Play Tennis Dataset (10 Rows, 6 Columns)  

| Outlook  | Temperature | Humidity | Wind  | Pressure | Play  |  
|----------|------------|----------|-------|----------|-------|  
| Sunny    | Hot        | High     | Weak  | Low      | No    |  
| Sunny    | Hot        | High     | Strong| Normal   | No    |  
| Overcast | Hot        | High     | Weak  | High     | Yes   |  
| Rain     | Mild       | High     | Weak  | Low      | Yes   |  
| Rain     | Cool       | Normal   | Weak  | Normal   | Yes   |  
| Rain     | Cool       | Normal   | Strong| High     | No    |  
| Overcast | Cool       | Normal   | Strong| Normal   | Yes   |  
| Sunny    | Mild       | High     | Weak  | High     | No    |  
| Sunny    | Cool       | Normal   | Weak  | Normal   | Yes   |  
| Rain     | Mild       | Normal   | Weak  | High     | Yes   |  

---

## Section 5: Entropy Calculation for Play Tennis  

Total Instances (N) = 10  
Yes = 6, No = 4  

H(Play) = - (6/10 log2(6/10)) - (4/10 log2(4/10))  
H(Play) = - (0.6 * -0.737) - (0.4 * -1.322)  
H(Play) = 0.971  

Entropy of Play = 0.971  

---

## Section 6: Information Gain Calculation for Each Attribute  

### 1. IG(Outlook)  

| Outlook  | Yes | No  | Total |  
|----------|----|----|------|  
| Sunny    | 1  | 3  | 4    |  
| Overcast | 2  | 0  | 2    |  
| Rain     | 3  | 1  | 4    |  

H(Sunny) = 0.811, H(Overcast) = 0, H(Rain) = 0.811  

IG(Outlook) = 0.971 - [0.4(0.811) + 0.2(0) + 0.4(0.811)]  
IG(Outlook) = 0.323  

### 2. IG(Temperature)  

| Temperature | Yes | No  | Total |  
|------------|----|----|------|  
| Hot        | 1  | 2  | 3    |  
| Mild       | 2  | 1  | 3    |  
| Cool       | 3  | 1  | 4    |  

IG(Temperature) = 0.097  

### 3. IG(Humidity)  

| Humidity | Yes | No  | Total |  
|----------|----|----|------|  
| High     | 2  | 3  | 5    |  
| Normal   | 4  | 1  | 5    |  

IG(Humidity) = 0.124  

### 4. IG(Wind)  

| Wind  | Yes | No  | Total |  
|-------|----|----|------|  
| Weak  | 5  | 1  | 6    |  
| Strong| 1  | 3  | 4    |  

IG(Wind) = 0.257  

---

## Section 7: Final Decision Tree  

Since IG(Outlook) = 0.323 is the highest, we use Outlook as the root.  

```
         Outlook  
        /   |   \  
     Sunny Overcast  Rain  
     /  \      |      /  \  
  Wind   No    Yes   Humidity  
  /   \         /  \  
Weak Strong  Normal  High  
Yes    No      Yes    No  
```  

Final Decision Tree is Built.  

---

## Section 8: Python Code for Decision Tree  

import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder  
from sklearn.tree import DecisionTreeClassifier, export_text  
from sklearn.metrics import accuracy_score  

df = pd.DataFrame("Sample.csv")  

le = LabelEncoder()  
for col in df.columns:  
    df[col] = le.fit_transform(df[col])  

X = df.iloc[:, :-1]  
y = df.iloc[:, -1]  
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
dt = DecisionTreeClassifier(criterion="entropy")  
dt.fit(x_train, y_train)  

tree_rules = export_text(dt, feature_names=list(X.columns))  
print(tree_rules)  

---

## Section 9: Advantages and Disadvantages  

### Advantages  
- Simple and easy to interpret.  
- Works with both categorical and numerical data.  
- No need for feature scaling.  

### Disadvantages  
- Prone to overfitting.  
- Unstable (small changes in data can alter the tree).  
- Not efficient for large datasets.  

Complete calculation and implementation are now provided in a well-structured manner.
