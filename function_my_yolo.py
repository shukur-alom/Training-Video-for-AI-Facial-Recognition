import yaml
import shutil
import random
import os
import cv2
import numpy as np



def find_current_folder_path(folder_name):
    """That Can Find any folder core path"""
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, folder_name)
    return folder_path


def make_file():
    """This function can make file that required for the system"""
    file_list = [
        "images",
        "anoted/images",
        "anoted/labels",
        "data/test/images",
        "data/test/labels",
        "data/train/images",
        "data/train/labels",
        "data/valid/images",
        "data/valid/labels",
    ]
    for h in file_list:
        try:
            os.makedirs(h)
        except:
            pass


def del_file():
    """This function can delete file that not required for the system"""

    for i in os.listdir("images"):
        try: os.rmdir(f"images/{i}")
        except:pass
    
    file_list = ["anoted/images", "anoted/labels", "anoted","images"]
    for h in file_list:
        try:
            os.rmdir(h)
        except:
            pass


def bnd_box_to_yolo_line(voc_bbox, img_size):
    """The Function convert CV2 box format to YOLO format.
    input CV2 box, and img shape"""
    x_min, y_min, x_max, y_max = voc_bbox

    for i in range(1, 60):
        try:
            x_min_, y_min_, x_max_, y_max_ = (
                x_min - i,
                y_min - i,
                x_max - (50 - i),
                y_max - (50 - i),
            )
        except:
            pass

    x_min, y_min, x_max, y_max = x_min_, y_min_, x_max_, y_max_

    x_center = (x_min + x_max) / (2 * img_size[1])
    y_center = (y_min + y_max) / (2 * img_size[0])
    width = (x_max - x_min) / img_size[1]
    height = (y_max - y_min) / img_size[0]
    return x_center, y_center, width, height


def make_yaml(list_name):
    """pass list like ['shukur','safwan'].
    it will update the yaml file that required for YOLO"""
    new_data = {
        "test": str(find_current_folder_path("data/test/images")),
        "train": str(find_current_folder_path("data/train/images")),
        "val": str(find_current_folder_path("data/valid/images")),
        "nc": len(list_name),
        "names": list_name,
    }
    with open("data.yaml", "r") as file:
        existing_data = yaml.safe_load(file)

    existing_data.update(new_data)

    with open("data.yaml", "w") as file:
        yaml.dump(existing_data, file, default_flow_style=False)


def move_file(from_, to_):
    """This function can move any file one folder to another folder
    inp (from_path,to_path)"""
    shutil.move(from_, to_)


def move_anoted_to_data(size, name):
    """This Function move image and text file from anoted folder to data folder randomly"""
    random_list = random.sample(range(0, len(os.listdir("anoted/images"))), size)
    img_path = os.listdir("anoted/images")
    for i in random_list:
        move_file(
            f"anoted/labels/{img_path[i].split('.')[-2]}.txt", f"data/{name}/labels"
        )
        move_file(f"anoted/images/{img_path[i]}", f"data/{name}/images")


def augment_image_1(original_image):
    aug_img_0 = cv2.flip(original_image, random.randint(-1, 1))
    blur_val = random.randint(3, 7)
    aug_img_1 = cv2.blur(original_image, (blur_val, blur_val))
    aug_img_2 = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    aug_img_3 = cv2.flip(original_image, -1)
    aug_img_5 = cv2.flip(original_image, 0)
    aug_img_4 = cv2.flip(original_image, 1)
    return aug_img_0, aug_img_1, aug_img_2, aug_img_3, aug_img_4, aug_img_5



def augment_image_2(img):
    """This Function van change your image brightness randomly"""
    value = random.randrange(-70, 70, 10)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, random.randrange(0, 180, 7), 1.0)
    img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    img = cv2.flip(img, random.randint(-1, 1))
    img = cv2.flip(img, random.randint(-1, 1))
    blur_val = random.randint(2, 7)
    img = cv2.blur(img, (blur_val, blur_val))

    return img



def vid_to_img():
  """This Function convert video to image"""
  for i in os.listdir("inp_video"):
      
      try:
        make_folder = f"images/{i.split('.')[0]}"
        os.makedirs(f"{make_folder}")
      except:pass

      cap = cv2.VideoCapture(f"inp_video/{i}")
      
      if (cap.isOpened()== False): 
        print("Error opening video stream or file")
      
      j  = 0
      print(" ")
      while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
      
          #Scv2.imshow(f"{i}",frame)
          print(">>",end="")
          cv2.imwrite(f"{make_folder}/@@!#!{j}.jpg", frame)
          j+=1
          if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        else: 
          break
    
      cap.release()
      
      cv2.destroyAllWindows()
