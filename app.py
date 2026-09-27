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

def draw_laser_dot(img, center, color, radius=12):
    overlay = img.copy()
    cv2.circle(overlay, center, radius + 8, color, -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    
    overlay2 = img.copy()
    cv2.circle(overlay2, center, radius, color, -1)
    cv2.addWeighted(overlay2, 0.7, img, 0.3, 0, img)
    
    cv2.circle(img, center, max(3, radius // 3), (255, 255, 255), -1)

def align_and_grade_40(image_bytes, keys_dict, total_q=40):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None, 0, [], "ไม่สามารถอ่านไฟล์รูปภาพได้ กรุณาถ่ายใหม่อีกครั้ง"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 6)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_boxes = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.75 <= aspect_ratio <= 1.25 and area > 120:
                    marker_boxes.append((x + w/2, y + h/2, area))

    marker_boxes = sorted(marker_boxes, key=lambda b: b[2], reverse=True)[:4]
    
    w_box, h_box = 800, 1100
    if len(marker_boxes) == 4:
        points = [(pt[0], pt[1]) for pt in marker_boxes]
        points = sorted(points, key=lambda p: p[1])
        top_two = sorted(points[:2], key=lambda p: p[0])
        bottom_two = sorted(points[-2:], key=lambda p: p[0])
        src_pts = np.float32([top_two[0], top_two[1], bottom_two[1], bottom_two[0]])
        dst_pts = np.float32([[25, 25], [w_box - 25, 25], [w_box - 25, h_box - 25], [25, h_box - 25]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_gray = cv2.warpPerspective(gray, matrix, (w_box, h_box))
        warped_vis = cv2.warpPerspective(image, matrix, (w_box, h_box))
    else:
        warped_gray = cv2.resize(gray, (w_box, h_box))
        warped_vis = cv2.resize(image, (w_box, h_box))

    # วาดกรอบเส้นเลเซอร์ HUD สีฟ้า
    cv2.rectangle(warped_vis, (25, 25), (w_box - 25, h_box - 25), (255, 255, 0), 2)
    cv2.line(warped_vis, (w_box // 2, 30), (w_box // 2, h_box - 30), (255, 255, 0), 1)

    cols_config = [
        {"x_start": 120, "x_end": 375, "q_range": (1, min(20, total_q))},
        {"x_start": 460, "x_end": 715, "q_range": (21, min(40, total_q))}
    ]
    
    y_start = 85
    y_step = 46.5
    choice_map = {0: "ก", 1: "ข", 2: "ค", 3: "ง"}
    choice_rev = {"ก": 0, "ข": 1, "ค": 2, "ง": 3}
    
    score = 0
    results = []

    for col in cols_config:
        q_start, q_end = col["q_range"]
        if q_start > total_q:
            continue

        c_w = (col["x_end"] - col["x_start"]) / 4.0
        for q_num in range(q_start, q_end + 1):
            row_idx = (q_num - 1) % 20
            cy = int(y_start + (row_idx * y_step) + 23)
            
            densities = []
            coords = []
            for c_idx in range(4):
                cx = int(col["x_start"] + (c_idx * c_w) + (c_w / 2))
                coords.append((cx, cy))
                
                mask = np.zeros(warped_gray.shape, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), 11, 255, -1)
                mean_val = cv2.mean(warped_gray, mask=mask)[0]
                densities.append(mean_val)

            best_idx = int(np.argmin(densities))
            darkest_val = densities[best_idx]
            sorted_dens = sorted(densities)
            margin = sorted_dens[1] - sorted_dens[0]

            student_ans = "ไม่ตอบ"
            if darkest_val < 170 and margin > 18:
                student_ans = choice_map[best_idx]

            key = keys_dict.get(str(q_num), "")
            is_correct = (student_ans == key and student_ans != "ไม่ตอบ")

            if is_correct:
                score += 1
                # เลเซอร์สีเขียวนีออน (ตอบถูก)
                draw_laser_dot(warped_vis, coords[best_idx], (0, 255, 0), radius=13)
            else:
                if student_ans != "ไม่ตอบ":
                    # เลเซอร์สีแดงนีออน (ตอบผิด)
                    draw_laser_dot(warped_vis, coords[best_idx], (0, 0, 255), radius=13)
                # วงเลเซอร์สีเหลืองชี้เฉลยที่ถูกต้อง
                if key in choice_rev:
                    correct_c_idx = choice_rev[key]
                    cv2.circle(warped_vis, coords[correct_c_idx], 14, (0, 230, 255), 2)

            results.append({
                "ข้อ": str(q_num),
                "คำตอบ": student_ans,
                "เฉลย": key,
                "ผล": "✔ ถูก" if is_correct else "✘ ผิด"
            })

    return warped_vis, score, results, None

st.set_page_config(page_title="ระบบตรวจข้อสอบ OMR เลเซอร์", layout="wide")
st.title("🎯 ระบบตรวจข้อสอบอัจฉริยะ (HUD Laser Scanner)")

db = load_db()
tab1, tab2, tab3 = st.tabs(["📷 สแกนตรวจข้อสอบ", "📊 ดูผลคะแนน / ดาวน์โหลด Excel", "⚙️ จัดการชุดเฉลย"])

with tab3:
    st.subheader("สร้างรายวิชาและเฉลย")
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
        st.info("ยังไม่มีข้อมูลรายวิชา กรุณาไปที่แท็บ '⚙️ จัดการชุดเฉลย' ด้านบนก่อน")
    else:
        subject_list = {f"{k} - {v['name']}": k for k, v in db.items()}
        selected = st.selectbox("เลือกวิชาที่จะตรวจ", list(subject_list.keys()))
        sub_info = db[subject_list[selected]]
        
        student_id = st.text_input("เลขที่ / รหัสนักเรียน", value="1")
        mode = st.radio("เลือกโหมดกล้อง:", ["📸 ถ่ายภาพสดด้วยกล้อง", "📁 อัปโหลดภาพจากเครื่อง"], horizontal=True)
        
        uploaded_file = None
        if mode == "📸 ถ่ายภาพสดด้วยกล้อง":
            uploaded_file = st.camera_input("ส่องกล้องให้เห็นสี่เหลี่ยมสีดำครบ 4 มุม แล้วกดถ่ายภาพ")
        else:
            uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพกระดาษคำตอบ", type=["jpg", "png", "jpeg"])

        if uploaded_file and st.button("⚡ ยิงเลเซอร์สแกนตรวจข้อสอบ"):
            with st.spinner("ระบบกำลังยิงเลเซอร์วิเคราะห์ตำแหน่งข้อสอบ..."):
                warped_vis, score, results, err_msg = align_and_grade_40(
                    uploaded_file.getvalue(), 
                    sub_info["keys"], 
                    sub_info["total"]
                )
                
                if err_msg:
                    st.error(err_msg)
                else:
                    st.success(f"✨ ผลการตรวจข้อสอบ: {score} / {sub_info['total']} คะแนน")
                    
                    col_hud, col_summary = st.columns([1.2, 0.8])
                    with col_hud:
                        st.markdown("**🔬 ภาพเลเซอร์วิเคราะห์คะแนน (HUD Laser Vision):**")
                        st.caption("🟢 จุดเลเซอร์เขียว = ถูก | 🔴 จุดเลเซอร์แดง = ผิด | 🟡 วงสีเหลือง = เฉลยข้อที่ถูก")
                        st.image(cv2.cvtColor(warped_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    with col_summary:
                        st.markdown("**📋 รายละเอียดคำตอบรายข้อ:**")
                        half = (len(results) + 1) // 2
                        r_col1, r_col2 = st.columns(2)
                        with r_col1: st.table(pd.DataFrame(results[:half]))
                        with r_col2: st.table(pd.DataFrame(results[half:]))
                    
                    new_row = {"เลขที่": student_id, "วิชา": sub_info["name"], "คะแนน": score, "เต็ม": sub_info["total"]}
                    if os.path.exists(RESULT_FILE):
                        df_all = pd.read_excel(RESULT_FILE)
                        df_all = pd.concat([df_all, pd.DataFrame([new_row])], ignore_index=True)
                    else:
                        df_all = pd.DataFrame([new_row])
                    df_all.to_excel(RESULT_FILE, index=False)
                    st.info(f"💾 บันทึกคะแนนของเลขที่ {student_id} ลงระบบเรียบร้อยแล้ว")

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
