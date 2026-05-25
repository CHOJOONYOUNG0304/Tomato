import streamlit as st
import pandas as pd
import joblib


# 모델 불러오기
# 예: rf_model.pkl 파일이 같은 폴더에 있어야 함
rf_model = joblib.load("tomato_model.pkl")

st.title("착과율 예측 프로그램")

st.write("내부온도, 내부습도, 지온을 입력하면 착과율을 예측합니다.")

# 입력값 받기
temp = st.number_input("내부온도 입력", value=25.0)
humidity = st.number_input("내부습도 입력", value=60.0)
soil_temp = st.number_input("지온 입력", value=20.0)

# 버튼 클릭 시 예측
if st.button("예측하기"):

    input_data = pd.DataFrame(
        [[temp, humidity, soil_temp]],
        columns=['내부온도', '내부습도', '지온']
    )

    # 모델 학습 시 사용한 컬럼 순서 맞추기
    input_data = input_data.reindex(
        columns=rf_model.feature_names_in_,
        fill_value=0
    )

    # 예측
    predicted = rf_model.predict(input_data)

    result = predicted.flatten()[0]

    st.success(f"예측 착과율 : {result:.1f}%")