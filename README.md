check_structure.py 我放置在 "D:\rcnn\A-HRNet\data\coco\annotations\check_structure.py"
mp_test.py 我放置在 "D:\rcnn\PyTorch-Object-Detection-Faster-RCNN-Tutorial\mp_test.py"
double_check.py 我放置在 "D:\rcnn\PyTorch-Object-Detection-Faster-RCNN-Tutorial\output_videos\double_check.py"
video_inference.inynb 我放置在 "D:\rcnn\PyTorch-Object-Detection-Faster-RCNN-Tutorial\video_inference.ipynb"
你們如果想用double_check.py 去檢查 bbox 是否在正確位置，可以先將 video_inference 的 if frame_count % 3 == 0: # 一樣每三幀取一次 改成 % 1，就可以了。之後正是要用再改回 % 3
