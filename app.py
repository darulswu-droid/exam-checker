import streamlit as st
import cv2
import numpy as np
import json
import os
import pandas as pd
from io import BytesIO

DB_FILE = "subjects.json"
RESULT_FILE = "exam_results.xlsx"

def load_db():
    if not os.path.exists(DB_FILE): return {}
    with open(DB_FILE, "r", encoding="utf-8") as f: return json.load(f)

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

def align_and_grade_40(image_bytes, total_q=40, num_choices=4):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # ตรวจหาสี่เหลี่ยมมาร์กเกอร์ 4 มุม
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_boxes = []
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.7 <= aspect_ratio <= 1.3 and cv2.contourArea(c) > 60:
                marker_boxes.append((x + w/2, y + h/2))

    # ดึงระนาบภาพให้ตรงตามมาร์กเกอร์
    w_box, h_box = 800, 1100
    if len(marker_boxes) >= 4:
        marker_boxes = sorted(marker_boxes, key=lambda p: p[1])
        top_two = sorted(marker_boxes[:2], key=lambda p: p[0])
        bottom_two = sorted(marker_boxes[-2:], key=lambda p: p[0])
        src_pts = np.float32([top_two[0], top_two[1], bottom_two[1], bottom_two[0]])
        dst_pts = np.float32([[0, 0], [w_box, 0], [w_box, h_box], [0, h_box]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(thresh, matrix, (w_box, h_box))
    else:
        warped = cv2.resize(thresh, (w_box, h_box))

    choice_map = {0: "ก", 1: "ข", 2: "ค", 3: "ง"}
    detected = []

    # กำหนดพิกัดกรอบตาราง 2 คอลัมน์ (ซ้าย: ข้อ 1-20, ขวา: ข้อ 21-40)
    cols_area = [
        {"x_start": int(w_box * 0.08), "x_end": int(w_box * 0.48), "q_range": (0, min(20, total_q))},
        {"x_start": int(w_box * 0.52), "x_end": int(w_box * 0.92), "q_range": (20, min(40, total_q))}
    ]
    
    y_start = int(h_box * 0.08)
    y_end = int(h_box * 0.92)
    rows_per_col = 20
    row_h = (y_end - y_start) // rows_per_col

    answers_dict = {}

    for area in cols_area:
        start_idx, end_idx = area["q_range"]
        col_w = (area["x_end"] - area["x_start"]) // (num_choices + 1) # เผื่อช่องเลขข้อ
        choices_start_x = area["x_start"] + col_w

        for i in range(start_idx, end_idx):
            row_idx = i - start_idx
            y1 = y_start + row_idx * row_h + int(row_h * 0.15)
            y2 = y_start + (row_idx + 1) * row_h - int(row_h * 0.15)

            densities = []
            for c in range(num_choices):
                x1 = choices_start_x + c * col_w + int(col_w * 0.15)
                x2 = choices_start_x + (c + 1) * col_w - int(col_w * 0.15)

                roi = warped[y1:y2, x1:x2]
                densities.append(cv2.countNonZero(roi))

            best = np.argmax(densities)
            threshold_val = (row_h * col_w * 0.045)
            if densities[best] > threshold_val:
                answers_dict[i + 1] = choice_map.get(best, "-")
            else:
                answers_dict[i + 1] = "ไม่ตอบ"

    for q in range(1, total_q + 1):
        detected.append(answers_dict.get(q, "ไม่ตอบ"))

    return detected

st.set_page_config(page_title="ระบบตรวจข้อสอบ OMR 40 ข้อ", layout="wide")
st.title("🎯 ระบบตรวจข้อสอบอัจฉริยะ (ฝน / กากบาท - 40 ข้อ)")

db = load_db()
tab1, tab2, tab3 = st.tabs(["📷 ตรวจกระดาษคำตอบ", "📊 ดูผลคะแนน / ดาวน์โหลด Excel", "⚙️ เพิ่ม/จัดการวิชาและเฉลย"])

with tab3:
    st.subheader("เพิ่มรายวิชาและชุดเฉลย")
    with st.form("add_form"):
        code = st.text_input("รหัสวิชา (เช่น SOC101)")
        name = st.text_input("ชื่อวิชา (เช่น สังคมศึกษา)")
        total_q = st.number_input("จำนวนข้อสอบ", min_value=1, max_value=40, value=40)
        
        st.write("ระบุเฉลยคำตอบแต่ละข้อ:")
        cols = st.columns(10)
        keys = {}
        for i in range(1, total_q + 1):
            with cols[(i - 1) % 10]:
                keys[str(i)] = st.selectbox(f"ข้อ {i}", ["ก", "ข", "ค", "ง"], key=f"ans_{i}")
        
        if st.form_submit_button("💾 บันทึกวิชา"):
            if code and name:
                db[code] = {"name": name, "total": total_q, "keys": keys}
                save_db(db)
                st.success(f"บันทึกวิชา {code} - {name} ({total_q} ข้อ) เรียบร้อยแล้ว!")
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
            detected = align_and_grade_40(uploaded_file.getvalue(), sub_info["total"])
            
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
            
            # แสดงผลแบบ 2 คอลัมน์ให้อ่านง่าย
            col_res1, col_res2 = st.columns(2)
            half = (len(results) + 1) // 2
            with col_res1:
                st.table(pd.DataFrame(results[:half]))
            with col_res2:
                st.table(pd.DataFrame(results[half:]))
            
            new_row = {"เลขที่": student_id, "วิชา": sub_info["name"], "คะแนน": score, "เต็ม": sub_info["total"]}
            if os.path.exists(RESULT_FILE):
                df_all = pd.read_excel(RESULT_FILE)
                df_all = pd.concat([df_all, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df_all = pd.DataFrame([new_row])
            df_all.to_excel(RESULT_FILE, index=False)
            st.success(f"บันทึกคะแนนของเลขที่ {student_id} เรียบร้อยแล้ว!")

with tab2:
    st.subheader("ตารางคะแนนรวมทั้งหมด")
    if os.path.exists(RESULT_FILE):
        df_display = pd.read_excel(RESULT_FILE)
        st.dataframe(df_display, use_container_width=True)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_display.to_excel(writer, index=False)
        excel_data = output.getvalue()
        
        st.download_button(
            label="📥 ดาวน์โหลดไฟล์ Excel (exam_results.xlsx)",
            data=excel_data,
            file_name="exam_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.write("ยังไม่มีข้อมูลคะแนนที่ตรวจ")