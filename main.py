from ultralytics import YOLO
import cv2 as cv
import os, time,random
from function_my_yolo import (
    bnd_box_to_yolo_line,
    make_yaml,
    move_file,
    move_anoted_to_data,
    augment_image_1,
    augment_image_2,
    make_file,
    del_file,
    vid_to_img
)


augmentation_size = int(input("Augmentation size : ")) #10
test_size =  int(input("Test size : "))/100  #0.18
val_size =  int(input("Val size : "))/100 #0.10



make_file()

vid_to_img()

model = YOLO("models/face model/weights/best.pt")

def cv2_box(img_path):
    img = cv.imread(img_path)
    results = model(img)
    for result in results:
        for i, data in enumerate(result.boxes.data.tolist()):
            confidence = data[4]
            if confidence>0.50:
                return bnd_box_to_yolo_line((data[0], data[1], data[2], data[3]), img.shape)


for co, data in enumerate(os.listdir("images")):
    folder_path = f"images/{data}"
    for g, i in enumerate(os.listdir(folder_path)):
        img_path = f"images/{data}/{i}"
        print(f"Image Augmentation {i}")
        img = cv.imread(img_path)
        img_0, img_1, img_2, img_3, img_4, img_5 = augment_image_1(img)
        cv.imwrite(f"images/{data}/{co}_{g}_img_0.jpg", img_0)
        cv.imwrite(f"images/{data}/{co}_{g}_img_1.jpg", img_1)
        cv.imwrite(f"images/{data}/{co}_{g}_img_2.jpg", img_2)
        cv.imwrite(f"images/{data}/{co}_{g}_img_3.jpg", img_3)
        cv.imwrite(f"images/{data}/{co}_{g}_img_4.jpg", img_4)
        cv.imwrite(f"images/{data}/{co}_{g}_img_5.jpg", img_5)
        try:
            for ghf in range(augmentation_size - 4):
                cv.imwrite(
                    f"images/{data}/{co}_{g}_{ghf}.jpg", augment_image_2(img)
                )
        except:
            print("Minimum 4 images")


make_yaml(os.listdir("images"))

count = 0
for co, data in enumerate(os.listdir("images")):
    folder_path = f"images/{data}"
    for i in os.listdir(folder_path):
        img_path = f"images/{data}/{i}"
        try:
            x, y, w, h = cv2_box(img_path)
            print(co, x, y, w, h)
            os.rename(img_path, f"images/{data}/{count}.{i.split('.')[-1]}")

            move_file(
                f"images/{data}/{count}.{i.split('.')[-1]}", "anoted/images"
            )

            f = open(f"anoted/labels/{count}.txt", "w+")
            f.write(f"{co} {x} {y} {w} {h}")
            count += 1

        except:
            os.rename(img_path, f"images/{data}/{count}.{i.split('.')[-1]}")
            move_file(
                f"images/{data}/{count}.{i.split('.')[-1]}", "anoted/images"
            )

            f = open(f"anoted/labels/{count}.txt", "w+")
            f.close()
            count += 1

model = 0
test = int(test_size * len(os.listdir("anoted/labels")))
valid = int(val_size * len(os.listdir("anoted/labels")))
train = len(os.listdir("anoted/labels")) - (test + valid)

move_anoted_to_data(test, "test")
move_anoted_to_data(valid, "valid")
move_anoted_to_data(train, "train")

del_file()