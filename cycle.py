from pathlib import Path
import pickle
import streamlit as st
import datetime
from PIL import Image
import pandas as pd
from datetime import timedelta

# ====================== главная страница ============================
# параметры главной страницы
# https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config
st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto",
    page_title="M-cycle",
    page_icon="🧊",
)
# ----------- функции -------------------------------------

# функция для загрузки картинки с диска
# кэшируем, иначе каждый раз будет загружаться заново
@st.cache_data
def load_image(image_path):
    image = Image.open(image_path)
    # обрезка до нужного размера с сохранением пропорций
    MAX_SIZE = (600, 400)
    image.thumbnail(MAX_SIZE)
    return image

# функция загрузки модели
# кэшируем, иначе каждый раз будет загружаться заново
@st.cache_data
def load_model(model_path):
    # загрузка сериализованной модели
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# ------------- загрузка картинки для страницы и модели ---------

# путь до картинки
image_path = Path.cwd() / 'Dias-fertiles-de-una-mujer.jpg'
image = load_image(image_path)

# путь до модели
model_path = Path.cwd() / 'model_sar.pkl'
model_sar = load_model(model_path)


# ---------- отрисовка текста и картинки ------------------------
st.write(
    """
    # Прогноз менструального цикла
    Введите ваши данные и получите результат
    """
)

# отрисовка картинки на странице
st.image(image)


# ====================== боковое меню для ввода данных ===============

st.sidebar.header('Введите данные')

# словарь с названиями признаков и описанием для удобства
features_dict = {
    'date_last': 'Дата последней менструации',
    'pred_num': 'Прогноз (количество месяцев)',
      }

# кнопки - слайдеры для ввода данных человека
date_last = st.sidebar.date_input(features_dict['date_last'], value=None)
pred_num = st.sidebar.slider(features_dict['pred_num'], min_value=1, max_value=3, value=1, step=1)


# предикт моделью входных данных, на выходе даты цикла
st_prediction = model_sar.forecast(pred_num)
st_prediction_int = st_prediction.astype(int)
pred_list = st_prediction_int.to_list()
new_list = []
for i in range(len(pred_list)):
    new_list.append(pred_list[i] + sum(pred_list[:i]))
pred_cicle = pd.DataFrame(new_list, columns=['sarima_pred'])
pred_cicle['sarima_pred'] = pred_cicle['sarima_pred'].astype(str)
pred_cicle['sarima_pred'] = pred_cicle['sarima_pred'] + ' days'
pred_cicle['sarima_pred'] = pd.to_timedelta(pred_cicle['sarima_pred'])
pred_cicle['Даты цикла'] = pd.Timestamp(date_last) + pred_cicle['sarima_pred']
pred_cicle["Дни овуляции"] = pred_cicle["Даты цикла"] - timedelta(days=14)
pred_cicle['a'] = '/'

# вывести предсказание модели
st.write("## Даты начала менструального цикла")
st.write(pred_cicle[['Даты цикла', 'a']].to_string(index=False, header=False))
st.write("## Дни овуляции")
st.write(pred_cicle[['Дни овуляции', 'a']].to_string(index=False, header=False))
