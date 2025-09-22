import json
import os

def super_gay(filename):

    if not os.path.exists(filename):
        print(f"錯誤：找不到檔案 '{filename}'")
        return
    
    with open(filename, 'r') as f:
        structure = json.load(f)

    print("檔案結構:\n")
    print(f" 最外層結構是: {type(structure).__name__}\n") 
    # type 是輸出<class ...> ，加 name 就可以輸出純名稱了
    print(f" 外層裡面包含的主要欄位有: {list(structure.keys())}\n")

    num_annotations = len(structure.get('annotations', []))
    print(f"總標註數量: {num_annotations:,}\n")

    if 'images' in structure and structure['images']:
        file_names = [img['file_name'] for img in structure['images']]
        max_filename = max(file_names)
        file_num = int(max_filename.split('.')[0])
        next_file_name = f"{file_num + 1:012d}.jpg" 
        print(f"file_name 最大值: {max_filename}，之後就從  {next_file_name} 這邊開始接")

    if 'annotations' in structure and structure['annotations']:
        max_ann_id = max(a['id'] for a in structure['annotations'])
        double_check_max_img = max(d['image_id'] for d in structure['annotations'])

        print(f"\n 標記人物的 ID 最大值: {max_ann_id}，可以從 {max_ann_id + 1} 繼續開始")
        print(f"  Annotations 中最大的 image_id: {double_check_max_img}，這個會代表有人類的圖片的最大編號，如果跟上面那個 file_name 最大值不一樣是正常的，可以自己去image資料夾檢查看看")


def format_json_file(original_file, formatted_file):

    if not os.path.exists(original_file):
        print(f"錯誤：找不到檔案 '{original_file}'")
        return

    with open(original_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"正在格式化內容 {formatted_file}...")
    with open(formatted_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("好了")


original_filepath = r'D:\rcnn\A-HRNet\data\coco\annotations\person_keypoints_train2017.json'
formatted_filepath = r'D:\rcnn\A-HRNet\data\coco\annotations\formatted_person_keypoints_train2017.json'

#if not os.path.exists(formatted_filepath):
#    format_json_file(original_filepath, formatted_filepath)
super_gay(original_filepath)
