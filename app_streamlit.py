import streamlit as st
import pandas as pd
import torch
import joblib
from model import MyModel  # твой класс MyModel
from scipy.special import inv_boxcox

# -----------------------------
# 1. Загрузка модели и скейлеров
# -----------------------------
device = torch.device("cpu")

cat_embs = [(8,4),(3,2),(3,2),(3,2),(2,1),(3,2)]
num_columns = ['Age', 'Experience', 'Security status', "Contribution to the company's security"]
categorical_columns = ['Post', 'The ability to perform security', 'There were security errors',
                       'There are risks at work', 'Participation in the audit', 'Interrupt work in case of danger']

model = MyModel(len(num_columns), cat_embs).to(device)
model.load_state_dict(torch.load("fully_neuro.pth", map_location=device))
model.eval()

scaler_dict = joblib.load('scalers.save')
encode_dict = joblib.load('encoders.save')
boxcox_lambda = 0.4245912219216052

# -----------------------------
# 2. Streamlit интерфейс
# -----------------------------
st.title("Оценка уровня КПБ")
st.write("Введите данные:")

# Числовые поля
age = st.number_input("Ваш возраст?", 18, 70, 30)
experience = st.number_input("Ваш общий стаж?", 0, 50, 5)
security_status = st.slider("Оцените, по шкале от 1 до 5, состояние производственной безопасности на Вашем рабочем месте", 1, 5, 3)
contribution = st.number_input("Как Вы оцениваете свой личный вклад в обеспечение и развитие безопасности своего структурного подразделения? (%)", 0, 100, 50)

# Категориальные поля (точно как в PyQt)
post = st.selectbox("Уровень занимаемой Вами должности", options=encode_dict['Post'].classes_)
ability_security = st.selectbox(
    "Считаю, что при выполнении работы я могу соблюдать все требования безопасности в рамках своей производственной деятельности",
    options=encode_dict['The ability to perform security'].classes_)
errors = st.selectbox(
    "Я не совершал ошибок в процессе своей производственной деятельности",
    options=encode_dict['There were security errors'].classes_)
risks = st.selectbox(
    "Я считаю, что в имеющихся условиях производственной деятельности возможно достичь показателя нулевого травматизма",
    options=encode_dict['There are risks at work'].classes_)
audit = st.selectbox(
    "Вы готовы сделать замечание коллеге в случае его небезопасных действий?",
    options=encode_dict['Interrupt work in case of danger'].classes_)
interrupt = st.selectbox(
    "Вы готовы остановить выполнение работы в случае предаварийной ситуации?",
    options=encode_dict['Interrupt work in case of danger'].classes_)

# -----------------------------
# 3. Предсказание
# -----------------------------
if st.button("Оценить уровень КПБ"):
    num_data = pd.DataFrame([[age, experience, security_status, contribution]], columns=num_columns)
    for col in num_columns:
        num_data[col] = scaler_dict[col].transform(num_data[col].values.reshape(-1,1))

    cat_data = pd.DataFrame([[post, ability_security, errors, risks, audit, interrupt]], columns=categorical_columns)
    for col in categorical_columns:
        cat_data[col] = encode_dict[col].transform(cat_data[col].values)

    num_tensor = torch.FloatTensor(num_data.values).to(device)
    cat_tensor = torch.LongTensor(cat_data.values).to(device)

    with torch.no_grad():
        y_transformed = model(num_tensor, cat_tensor).cpu().numpy()[0][0]
        y_real = inv_boxcox(y_transformed, boxcox_lambda) - 1

    st.success(f"Уровень КПБ: {y_real:.2f}")
