from datetime import datetime, timedelta

import cv2
import numpy as np
import math
from ultralytics import YOLO
import time


def calc_iou(x1, y1, x2, y2, x3, y3, x4, y4):
    x_inner1 = max(x1, x3)
    y_inner1 = max(y1, y3)
    x_inner2 = min(x2, x4)
    y_inner2 = min(y2, y4)
    width_inner = abs(x_inner2 - x_inner1)
    height_inner = abs(y_inner2 - y_inner1)
    area_inter = width_inner * height_inner
    width_box1 = abs(x1 - x2)
    height_box1 = abs(y1 - y2)
    width_box2 = abs(x3 - x4)
    height_box2 = abs(y3 - y4)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union
    return iou


def main():
    model_for_detect_heads = YOLO('best_yolov8m.pt')
    model_for_detect_persons = YOLO('yolov8l.pt')

    cap = cv2.VideoCapture('data/Input/Video2.mp4')
    ret, frame = cap.read()

    cap_write = cv2.VideoWriter('data/Output/Video2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                                (frame.shape[1], frame.shape[0]))

    # fps_start_time = time.time()
    fps = 0
    fps_end_time = 0

    person_table_and_time_sitting = {}
    tables_xyxy = {}
    start_time = datetime(2023, 1, 1, 0, 0, 0)

    results_of_person_det = model_for_detect_persons.track(frame, classes=[0, 60], persist=True,
                                                           tracker="bytetrack.yaml")

    for result in results_of_person_det[0].boxes.data.tolist():
        x1, y1, x2, y2, id, score, class_id = result

        if class_id == 60:
            tables_xyxy[id] = (int(x1), int(y1), int(x2), int(y2))
        else:
            continue

    for result in results_of_person_det[0].boxes.data.tolist():
        x1, y1, x2, y2, id, score, class_id = result

        if class_id == 1:
            for key, xyxy in tables_xyxy.items():
                x1_1, y1_1, x2_1, y2_1 = xyxy
                iou = calc_iou(x1, y1, x2, y2, x1_1, y1_1, x2_1, y2_1)
                if 0 < iou <= 1:
                    person_table_and_time_sitting[id] = (key, start_time)
                    break
        else:
            continue

    while ret:
        for result in results_of_person_det[0].boxes.data.tolist():

            check_head = True

            x1, y1, x2, y2, id, score, class_id = result
            temp_img = frame[int(y1):int(y2), int(x1):int(x2)]

            if class_id == 0:
                time = start_time
                if id in person_table_and_time_sitting:
                    id_table, time = person_table_and_time_sitting[id]
                    x1_1, y1_1, x2_1, y2_1 = tables_xyxy[id_table]
                    iou = calc_iou(x1, y1, x2, y2, x1_1, y1_1, x2_1, y2_1)

                    if 0 < iou <= 1:
                        time += timedelta(seconds=1)
                        person_table_and_time_sitting[id] = (id_table, time)
                    else:
                        del person_table_and_time_sitting[id]

                else:
                    for key, xyxy in tables_xyxy.items():
                        x1_1, y1_1, x2_1, y2_1 = xyxy
                        iou = calc_iou(x1, y1, x2, y2, x1_1, y1_1, x2_1, y2_1)
                        if 0 < iou <= 1:
                            person_table_and_time_sitting[id] = (key, start_time)
                            break

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=2)
                cv2.putText(frame, f'{time.time()}', (int(x1), int(y1)), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 139), thickness=2)
                cv2.putText(frame, f'table{id:.0f}', (int(x1), int(y1)), cv2.FONT_ITALIC, 1, (255, 0, 139), 2)
                check_head = False

            if check_head:
                results_of_head_det = model_for_detect_heads.predict(temp_img)
                for head_result in results_of_head_det[0].boxes.xyxy.tolist():
                    x1_1, y1_1, x2_1, y2_1, = head_result
                    cv2.rectangle(frame, (int(x1_1 + x1), int(y1_1 + y1)), (int(x2_1 + x1), int(y2_1 + y1)),
                                  (255, 0, 0),
                                  thickness=2)

        # fps_end_time = time.time()
        # time_diff = fps_end_time - fps_start_time
        # fps = 1 / time_diff
        # fps_text = f"FPS: {fps:.2f}"
        #
        # cv2.putText(frame, fps_text, (0, int(frame.shape[0] * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow('res', frame)

        ret, frame = cap.read()

        if ret:
            results_of_person_det = model_for_detect_persons.track(frame, classes=[0, 60], persist=True,
                                                                   tracker="bytetrack.yaml")

        fps_start_time = time.time()
        cv2.waitKey(1)

    cap.release()
    cap_write.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
