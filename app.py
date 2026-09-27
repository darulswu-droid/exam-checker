import streamlit as st
import cv2
import numpy as np
import json
import os
import pandas as pd

DB_FILE = "subjects.json"
RESULT_FILE = "exam_results.xlsx"

def load_db():
    if not os.path.exists(DB_FILE): return {}
    with open(DB_FILE, "r", encoding="utf-8") as f: return json.load(f)

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

def align_and_grade(image_bytes, num_q, num_choices=4):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_boxes = []
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.7 <= aspect_ratio <= 1.3 and cv2.contourArea(c) > 80:
                marker_boxes.append((x + w/2, y + h/2))

    warped = thresh
    if len(marker_boxes) >= 4:
        marker_boxes = sorted(marker_boxes, key=lambda p: p[1])
        top_two = sorted(marker_boxes[:2], key=lambda p: p[0])
        bottom_two = sorted(marker_boxes[-2:], key=lambda p: p[0])
        src_pts = np.float32([top_two[0], top_two[1], bottom_two[1], bottom_two[0]])

        w_box, h_box = 500, 700
        dst_pts = np.float32([[0, 0], [w_box, 0], [w_box, h_box], [0, h_box]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(thresh, matrix, (w_box, h_box))

    h, w = warped.shape
    row_h = (h - 140) // num_q
    start_y = 100
    col_w = w // num_choices

    choice_map = {0: "ก", 1: "ข", 2: "ค", 3: "ง"}
    detected = []

    for q in range(num_q):
        densities = []
        for c in range(num_choices):
            y1 = start_y + q * row_h + int(row_h * 0.1)
            y2 = start_y + (q + 1) * row_h - int(row_h * 0.1)
            x1 = c * col_w + int(col_w * 0.2)
            x2 = (c + 1) * col_w - int(col_w * 0.2)

            roi = warped[y1:y2, x1:x2]
            densities.append(cv2.countNonZero(roi))

        best = np.argmax(densities)
        if densities[best] > (row_h * col_w * 0.04):
            detected.append(choice_map.get(best, "-"))
        else:
            detected.append("ไม่ตอบ")

    return detected

st.set_page_config(page_title="ระบบตรวจข้อสอบ OMR", layout="wide")
st.title("🎯 ระบบตรวจข้อสอบอัจฉริยะ (ฝน / กากบาท)")

db = load_db()
tab1, tab2 = st.tabs(["📷 ตรวจกระดาษคำตอบ", "⚙️ เพิ่ม/จัดการวิชาและเฉลย"])

with tab2:
    st.subheader("เพิ่มรายวิชาและชุดเฉลย")
    with st.form("add_form"):
        code = st.text_input("รหัสวิชา (เช่น SOC101)")
        name = st.text_input("ชื่อวิชา (เช่น สังคมศึกษา ป.5)")
        total_q = st.number_input("จำนวนข้อสอบ", min_value=1, max_value=50, value=10)
        
        st.write("ระบุเฉลยคำตอบแต่ละข้อ:")
        cols = st.columns(5)
        keys = {}
        for i in range(1, total_q + 1):
            with cols[(i - 1) % 5]:
                keys[str(i)] = st.selectbox(f"ข้อ {i}", ["ก", "ข", "ค", "ง"], key=f"ans_{i}")
        
        if st.form_submit_button("💾 บันทึกวิชา"):
            if code and name:
                db[code] = {"name": name, "total": total_q, "keys": keys}
                save_db(db)
                st.success(f"บันทึกวิชา {code} - {name} เรียบร้อยแล้ว!")
            else:
                st.error("กรุณากรอกรหัสและชื่อวิชาให้ครบถ้วน")

with tab1:
    if not db:
        st.info("ยังไม่มีข้อมูลรายวิชา กรุณาไปที่แท็บ '⚙️ เพิ่ม/จัดการวิชาและเฉลย' ด้านบนก่อน")
    else:
        subject_list = {f"{k} - {v['name']}": k for k, v in db.items()}
        selected = st.selectbox("เลือกวิชาที่จะตรวจ", list(subject_list.keys()))
        sub_info = db[subject_list[selected]]
        
        student_id = st.text_input("เลขที่ / รหัสนักเรียน", value="1")
        uploaded_file = st.file_uploader("อัปโหลดหรือถ่ายภาพกระดาษคำตอบ", type=["jpg", "png", "jpeg"])

        if uploaded_file and st.button("🚀 เริ่มตรวจข้อสอบ"):
            detected = align_and_grade(uploaded_file.getvalue(), sub_info["total"])
            
            score = 0
            results = []
            for i in range(sub_info["total"]):
                q = str(i + 1)
                ans = detected[i]
                key = sub_info["keys"].get(q)
                is_correct = (ans == key)
                if is_correct: score += 1
                results.append({"ข้อ": q, "คำตอบ": ans, "เฉลย": key, "ผล": "✔ ถูก" if is_correct else "✘ ผิด"})

            st.subheader(f"ผลคะแนน: {score} / {sub_info['total']} คะแนน")
            st.table(pd.DataFrame(results))
            
            new_row = {"เลขที่": student_id, "วิชา": sub_info["name"], "คะแนน": score, "เต็ม": sub_info["total"]}
            if os.path.exists(RESULT_FILE):
                df_all = pd.read_excel(RESULT_FILE)
                df_all = pd.concat([df_all, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df_all = pd.DataFrame([new_row])
            df_all.to_excel(RESULT_FILE, index=False)
            st.success(f"บันทึกคะแนนของเลขที่ {student_id} ลงไฟล์ Excel ({RESULT_FILE}) เรียบร้อยแล้ว!")