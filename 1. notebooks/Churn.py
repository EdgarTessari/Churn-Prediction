# -*- coding: utf-8 -*-
#   1. Leitura dos dados


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Churn-Prediction/2. data/HR-Employee-Attrition.csv')

print(df.head())
print("\nInformações do DataFrame:")
print(df.info())
print("\nEstatísticas descritivas:")
print(df.describe())
print("\nValores nulos por coluna:")
print(df.isnull().sum())

#%% 2. Gráficos iniciais (Análise Exploratória)


sns.set(style="whitegrid")

plt.figure(figsize=(6,4))
ax = sns.countplot(data=df, x='Attrition', palette='Set2')
plt.title('Employee TurnOver')
plt.xlabel('Attrition (Saída)')
plt.ylabel('Qtt')
for p in ax.patches:
    count = p.get_height()
    percentage = 100 * count / len(df)
    ax.annotate(f'{count:.0f} ({percentage:.1f}%)', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 4),
                textcoords='offset points')
plt.show()

plt.figure(figsize=(10,6))
ax = sns.histplot(df['Age'], kde=True, color='skyblue')
plt.title("Employee's Age Distribution")
plt.xlabel('Age')
plt.ylabel('Qtt')
for p in ax.patches:
    count = p.get_height()
    if count > 0:
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(f'{int(count)}', (x, y), 
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 8),
                    textcoords='offset points')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, palette='coolwarm')
plt.title('Salário Mensal por Status de Saída')
plt.show()

#%% 3. Limpeza e Tratamento de Dados


print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

df = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'])

#%% 4. Análise Exploratória de Dados (EDA)


plt.figure(figsize=(7,5))
ax = sns.countplot(data=df, x='Gender', hue='Attrition', palette='Set2')
plt.title('Gender vs Attrition')
plt.xlabel('Gender')
plt.ylabel('Qtt')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.legend(title='Attrition')
plt.show()

plt.figure(figsize=(9,5))
ax = sns.countplot(data=df, x='Department', hue='Attrition', palette='Set2')
plt.title('Department vs Attrition')
plt.xlabel('Department')
plt.ylabel('Qtt')
plt.xticks(rotation=15)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.legend(title='Attrition')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Attrition', y='DistanceFromHome', data=df, palette='Set2')
plt.title('Distância de Casa por Status de Saída')
plt.xlabel('Attrition')
plt.ylabel('Distância de Casa (km)')
plt.show()

plt.figure(figsize=(6,4))
ax = sns.countplot(x='OverTime', hue='Attrition', data=df, palette='Set2')
plt.title('OverTime vs Attrition')
plt.xlabel('OverTime')
plt.ylabel('Qtt')
plt.legend(title='Attrition')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', fontsize=9, color='black', 
                xytext=(0, 6), textcoords='offset points')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
ax = sns.countplot(x='JobSatisfaction', hue='Attrition', data=df, palette='Set2')
plt.title('Satisfação no Trabalho vs Attrition')
plt.xlabel('Satisfaction Rating (1-4)')
plt.ylabel('Qtt')
plt.legend(title='Attrition')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=9, color='black',
                xytext=(0, 6), textcoords='offset points')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
ax = sns.countplot(y='JobRole', hue='Attrition', data=df, palette='Set2')
plt.title('Cargo vs Attrition')
plt.xlabel('Qtt')
plt.ylabel('Job Role')
plt.legend(title='Attrition')
for p in ax.patches:
    ax.annotate(f'{p.get_width()}',
                (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2),
                ha='left', va='center', fontsize=9, color='black',
                xytext=(4, 0), textcoords='offset points')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Attrition', y='YearsAtCompany', data=df, palette='Set2')
plt.title('Anos na Empresa vs Attrition')
plt.xlabel('Attrition')
plt.ylabel('Years At Company')
plt.tight_layout()
plt.show()

plt.figure(figsize=(16,12))
correlation_matrix = df.select_dtypes(include='number').corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Mapa de Calor das Correlações')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='TotalWorkingYears', y='MonthlyIncome', hue='Attrition', data=df, palette='Set2')
plt.title('MonthlyIncome vs TotalWorkingYears')
plt.xlabel('Total de Anos Trabalhados')
plt.ylabel('Salário Mensal')
plt.legend(title='Attrition')
plt.tight_layout()
plt.show()


#%% 5. Modelagem Preditiva - Regressão Logística


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df_model = df.copy()

df_model['Attrition'] = df_model['Attrition'].map({'Yes': 1, 'No': 0})

le = LabelEncoder()
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop('Attrition', axis=1)
y = df_model['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print(f"\nAcurácia: {accuracy_score(y_test, y_pred)*100:.2f}%")


from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


X = df.drop('Attrition', axis=1)
y = df['Attrition'].map({'Yes': 1, 'No': 0})

X_encoded = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

logreg_bal = LogisticRegression(max_iter=1000, random_state=42)
logreg_bal.fit(X_train_res, y_train_res)

y_pred_bal = logreg_bal.predict(X_test)

print("Matriz de Confusão com SMOTE:")
print(confusion_matrix(y_test, y_pred_bal))
print("\nRelatório de Classificação com SMOTE:")
print(classification_report(y_test, y_pred_bal))
print(f"Acurácia: {accuracy_score(y_test, y_pred_bal) * 100:.2f}%")


#%% 6. Modelagem Preditiva - Árvore de Decisão


from sklearn.tree import DecisionTreeClassifier

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_res_scaled, y_res)

X_test_scaled = scaler.transform(X_test)

y_pred = dt_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Matriz de Confusão:")
print(cm)
print("\nRelatório de Classificação:")
print(cr)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia: {accuracy * 100:.2f}%")


X = df.drop('Attrition', axis=1)
y = df['Attrition'].map({'Yes': 1, 'No': 0})

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)




dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train_res)

y_pred_dt = dt_model.predict(X_test_scaled)

print("Matriz de Confusão:")
cm = confusion_matrix(y_test, y_pred_dt)
print(cm)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_dt))
print("Acurácia:", round(accuracy_score(y_test, y_pred_dt)*100, 2), "%")


plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Set2', xticklabels=['Não', 'Sim'], yticklabels=['Não', 'Sim'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Árvore de Decisão')
plt.show()


#%% 7. Modelagem Preditiva - Random Forest


from sklearn.ensemble import RandomForestClassifier

X = df.drop('Attrition', axis=1)
y = df['Attrition'].map({'Yes': 1, 'No': 0})

X = pd.get_dummies(X, drop_first=True)

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")



scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)


X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_res_scaled, y_res, test_size=0.3, random_state=42, stratify=y_res )


from sklearn.model_selection import GridSearchCV


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


rf = RandomForestClassifier(random_state=42)


grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='accuracy')

grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test_scaled)

print("Melhores hiperparâmetros:", grid_search.best_params_)
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print(f"\nAcurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")


#%% 8. Modelagem Preditiva - XGBoost


from xgboost import XGBClassifier


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_res_scaled, y_train_res)

y_pred_xgb = xgb_model.predict(X_test_scaled)

conf_matrix = confusion_matrix(y_test, y_pred_xgb)
class_report = classification_report(y_test, y_pred_xgb)
accuracy = accuracy_score(y_test, y_pred_xgb)

print("Matriz de Confusão:")
print(conf_matrix)
print("\nRelatório de Classificação:")
print(class_report)
print(f"Acurácia: {accuracy * 100:.2f}%")


from xgboost import plot_importance


plt.figure(figsize=(10, 6))
plot_importance(xgb_model, importance_type='gain', max_num_features=15, title='Importância das Variáveis (Gain)', xlabel='Ganho')
plt.tight_layout()
plt.show()

feature_names = X.columns
importances = xgb_model.feature_importances_

sorted_idx = importances.argsort()[::-1]

for idx in sorted_idx[:5]:
    print(f"{feature_names[idx]}: {importances[idx] * 100:.2f}")
    

feature_names = X.columns

importances = xgb_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances * 100
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Top 15 variáveis mais importantes para o XGBoost:")
print(feature_importance_df.head(15))

top_features = {
    "Feature": [
        "StockOptionLevel", "Department_Sales", "Department_Research & Development",
        "JobRole_Sales Representative", "JobRole_Human Resources", "JobLevel",
        "EducationField_Other", "JobInvolvement", "JobRole_Laboratory Technician",
        "MaritalStatus_Married", "JobRole_Manager", "RelationshipSatisfaction",
        "JobSatisfaction", "EducationField_Medical", "BusinessTravel_Travel_Rarely"
    ],
    "Importance": [
        8.367497, 8.114747, 7.148386, 6.369535, 6.262721, 5.690520, 3.444334,
        3.192836, 3.006486, 2.925004, 2.541763, 2.438782, 2.352661, 2.336221, 2.213541
    ]
}

df_importance = pd.DataFrame(top_features)
df_importance.sort_values(by="Importance", ascending=False, inplace=True)

plt.figure(figsize=(10, 8))
barplot = sns.barplot(
    x="Importance", y="Feature", data=df_importance, color="cornflowerblue"
)
plt.title("Top 15 Most Important Variables (XGBoost - Gain)", fontsize=14)
plt.xlabel("Gain (%)")
plt.ylabel("Variable")

for index, value in enumerate(df_importance["Importance"]):
    plt.text(value + 0.1, index, f"{value:.2f}", va='center')

plt.tight_layout()
plt.show()


cm = np.array([[343, 27],
               [32, 338]])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='GnBu', cbar=False, 
            xticklabels=['Permaneceu', 'Saiu'], 
            yticklabels=['Permaneceu', 'Saiu'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Matriz de Confusão - XGBoost')
plt.tight_layout()
plt.show()